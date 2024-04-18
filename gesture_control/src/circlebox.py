#!/usr/bin/env python3

from __future__ import print_function


# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

from point2d import Point2D
import alvinxy.alvinxy as axy # Import AlvinXY transformation module
from geometry_msgs.msg import PoseStamped, PoseArray, Pose

# GEM Sensor Headers
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva

# GEM PACMod Headers
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# ROS Headers
import rospy

# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64

from sensor_msgs.msg import PointCloud2

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd

# turn commands
RIGHT = 0
STRAIGHT = 1
LEFT = 2

# threshold for when we consider ourselves close enough to the center point
DELTA = 1.5
# steering angle for making a circle
CIRCLE_STEERING_ANGLE = 20

class PID(object):

    def __init__(self, kp, ki, kd, wg=None):

        self.iterm  = 0
        self.last_t = None
        self.last_e = 0
        self.kp     = kp
        self.ki     = ki
        self.kd     = kd
        self.wg     = wg
        self.derror = 0

    def reset(self):
        self.iterm  = 0
        self.last_e = 0
        self.last_t = None

    def get_control(self, t, e, fwd=0):

        if self.last_t is None:
            self.last_t = t
            de = 0
        else:
            de = (e - self.last_e) / (t - self.last_t)

        if abs(e - self.last_e) > 0.5:
            de = 0

        self.iterm += e * (t - self.last_t)

        # take care of integral winding-up
        if self.wg is not None:
            if self.iterm > self.wg:
                self.iterm = self.wg
            elif self.iterm < -self.wg:
                self.iterm = -self.wg

        self.last_e = e
        self.last_t = t
        self.derror = de

        return fwd + self.kp * e + self.ki * self.iterm + self.kd * de


class OnlineFilter(object):

    def __init__(self, cutoff, fs, order):
        
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Get the filter coefficients 
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Initialize
        self.z = signal.lfilter_zi(self.b, self.a)
    
    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted


class CircleBox(object):
    
    def __init__(self):

        

        self.driving_to_center = False
        self.drove_to_center = False

        self.circle_made = False
        self.making_circle = False
        self.half_circle_first = True # print once we have made a half circle

        self.radius_set = False
        self.make_figure_8 = False
        self.midpoint_reached = False

        self.rate       = rospy.Rate(10)
        self.log = open('log3.txt', 'a')

        self.look_ahead = 4
        self.wheelbase  = 1.75 # meters
        self.offset     = 0.46 # meters'

        self.steering_angle = CIRCLE_STEERING_ANGLE


        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

        self.speed_sub  = rospy.Subscriber("/pacmod/as_tx/vehicle_speed", Float64, self.speed_callback)
        self.speed      = 0.0

    
        self.desired_speed = 0.4  # m/s, reference speed
        self.max_accel     = 0.4 # % of acceleration
        self.pid_speed     = PID(1.5, 0.3, 0.6, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

        # -------------------- PACMod setup --------------------

        self.gem_enable    = False
        self.pacmod_enable = False

        # GEM vehicle gps subscribe
        self.gnss_sub   = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        self.lat        = 0.0
        self.lon        = 0.0
        self.heading    = 0.0
        self.olat       = 40.0928563 # original lat/long # TODO, need to find where this is physically and start car here
        self.olon       = -88.2359994

        # GEM vehicle enable, publish once
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        # GEM vechile turn signal control
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 2 * np.pi - 0.01 # radians/second

        # midpoint callback
        self.midpoint_sub = rospy.Subscriber("midpoint", PoseArray, self.midpoint_callback)

        self.midpoint = []
        self.box_centroids = (None, None)


    def inspva_callback(self, inspva_msg):
        self.lat     = inspva_msg.latitude  # latitude
        self.lon     = inspva_msg.longitude # longitude
        self.heading = inspva_msg.azimuth   # heading in degrees # from where? x-axis(east)?

    # gets the midpoint from the lidar node
    def midpoint_callback(self, msg):
        if not self.radius_set:
            self.midpoint = [msg.poses[2].position.x, msg.poses[2].position.y, 0]
            self.box_centroids = ((msg.poses[0].position.x, msg.poses[0].position.y, 0),
                                (msg.poses[1].position.x, msg.poses[1].position.y, 0))
            print("Midpoint: ", self.midpoint)
            print("Box0: ", self.box_centroids[0])
            print("Box1: ", self.box_centroids[1])

    def speed_callback(self, msg):
        self.speed = round(msg.data, 3) # forward velocity in m/s

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def get_circle_time(self, radius):
        circumference = 2 * math.pi * radius
        return circumference / self.desired_speed

    def front2steer(self, f_angle):

        if(f_angle > 35):
            f_angle = 35

        if (f_angle < -35):
            f_angle = -35

        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)

        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0

        # print("steer_angle: " + str(steer_angle))
        return steer_angle
    
    def maintain_speed(self):
        # disengage brake
        self.brake_cmd.f64_cmd = 0
        self.brake_pub.publish(self.brake_cmd)

        # get output acceleration from filter+PID
        filt_vel     = self.speed_filter.get_data(self.speed)
        output_accel = self.pid_speed.get_control(rospy.get_time(), self.desired_speed - filt_vel)
        # bound accel values
        if output_accel > self.max_accel:
            output_accel = self.max_accel

        if output_accel < 0.3:
            output_accel = 0.3

        # to do: calculate angle based on radius
        self.accel_cmd.f64_cmd = output_accel
        self.accel_pub.publish(self.accel_cmd)
        # print("publishing accel:", self.accel_cmd.f64_cmd)

    # def make_figure_8(self, steering_direction, turn_signal):


    def make_circle(self, steering_direction, turn_signal_direction):
        if(not self.making_circle and not self.circle_made):
            self.making_circle = True
            self.circle_duration = self.get_circle_time(self.radius)
            self.start_time = rospy.Time.now()
            print("circle duration:", self.circle_duration)
        else:
            self.turn_cmd.ui16_cmd = turn_signal_direction
            self.turn_pub.publish(self.turn_cmd)
            self.steer_cmd.angular_position = np.radians(self.front2steer(steering_direction))
            self.steer_pub.publish(self.steer_cmd)

            curr_x, curr_y, curr_yaw = self.get_gem_state()
            p1 = [curr_x, curr_y]

            # stop if we've gone on for long enough (approx half circle) and have reached center
            elapsed_time = rospy.Time.now() - self.start_time

            if(elapsed_time.to_sec() > self.circle_duration / 2 and self.half_circle_first):
                print("made half circle")
                self.half_circle_first = False
            

            distance = dist(p1, self.start_point) 
            print("p1", p1)
            print("midpoint:", self.midpoint[0:2])
            print("startpoint:", self.start_point)
            print("distance from car to startpoint:", distance, ", DELTA:", DELTA)
            if(elapsed_time.to_sec() > self.circle_duration / 4 and distance < DELTA):
                if not self.make_figure_8:
                    print("Done making circle")
                    self.circle_made = True
                else:
                    print("Done making first circle")
                    self.steering_angle = -self.steering_angle
                    self.start_point = [self.curr_x, self.curr_y]
                    self.making_circle = False
                    self.half_circle_first = True
                    self.make_figure_8 = False

    def make_quarter_circle(self, steering_direction, turn_signal_direction):

        self.turn_cmd.ui16_cmd = turn_signal_direction
        self.turn_pub.publish(self.turn_cmd)
        self.steer_cmd.angular_position = np.radians(self.front2steer(steering_direction))
        self.steer_pub.publish(self.steer_cmd)

        curr_x, curr_y, curr_yaw = self.get_gem_state()
        p1 = [curr_x, curr_y]

        distance = dist(p1, self.midpoint) 
        print("p1", p1)
        print("midpoint:", self.midpoint[0:2])
        # print("startpoint:", self.start_point)
        print("distance from car to startpoint:", distance, ", DELTA:", 1.5)
        if(distance < 1.5):
            print("Done making quarter circle")
            self.midpoint_reached = True
            self.start_point = [curr_x, curr_y]

    def stop(self):
        print("stopping")
        self.turn_cmd.ui16_cmd = STRAIGHT
        self.turn_pub.publish(self.turn_cmd)
        self.steer_cmd.angular_position = 0
        self.steer_pub.publish(self.steer_cmd)
        self.accel_cmd.f64_cmd = 0
        self.accel_pub.publish(self.accel_cmd)
        self.brake_cmd.f64_cmd = 1.0
        self.brake_pub.publish(self.brake_cmd)

    def wps_to_local_xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
        return lon_wp_x, lat_wp_y   

    def get_gem_state(self):

        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # reference point is located at the center of GNSS antennas
        local_x_curr, local_y_curr = self.wps_to_local_xy(self.lon, self.lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = self.heading_to_yaw(self.heading) 

        # reference point is located at the center of rear axle
        curr_x = local_x_curr - self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr - self.offset * np.sin(curr_yaw)

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

    # find the angle bewtween two vectors    
    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        # [-pi, pi]
        return np.arctan2(sinang, cosang)
    
    def get_angle_between_points(self, p1, p2):
      return np.arctan2((p1[1] - p2[1]),(p1[0] - p2[0]))

    def start_circlebox(self):
        
        while not rospy.is_shutdown():

            if (self.gem_enable == False):

                if(self.pacmod_enable == True):

                    # ---------- enable PACMod ----------

                    # enable forward gear
                    self.gear_cmd.ui16_cmd = 3

                    # enable brake
                    self.brake_cmd.enable  = True
                    self.brake_cmd.clear   = False
                    self.brake_cmd.ignore  = False
                    self.brake_cmd.f64_cmd = 0.0

                    # enable gas 
                    self.accel_cmd.enable  = True
                    self.accel_cmd.clear   = False
                    self.accel_cmd.ignore  = False
                    self.accel_cmd.f64_cmd = 0.0

                    self.gear_pub.publish(self.gear_cmd)
                    print("Foward Engaged!")

                    self.turn_pub.publish(self.turn_cmd)
                    print("Turn Signal Ready!")
                    
                    self.brake_pub.publish(self.brake_cmd)
                    print("Brake Engaged!")

                    self.accel_pub.publish(self.accel_cmd)
                    print("Gas Engaged!")

                    self.gem_enable = True
            curr_x, curr_y, curr_yaw = self.get_gem_state()
            self.curr_x, self.curr_y, self.curr_yaw = curr_x, curr_y, curr_yaw

            # maintain fixed velocity
            if self.midpoint is not None and self.box_centroids[0] is not None and self.box_centroids[1] is not None:
                if not self.radius_set:
                    self.radius = dist(self.midpoint[0:2], self.box_centroids[0][0:2])
                    self.radius_set = True

                    # self.midpoint[0] += curr_x
                    # self.midpoint[1] += curr_y
                    
                    # bad circles
                    self.midpoint[0] = curr_x - self.midpoint[0]
                    self.midpoint[1] = curr_y - self.midpoint[1]

                    # self.start_point = [curr_x, curr_y]

                if self.midpoint_reached:
                    if not self.circle_made:
                        self.maintain_speed()
                        self.make_circle(self.steering_angle, LEFT)
                    else:
                        print("Publishing brake")
                        self.accel_cmd.f64_cmd = 0
                        self.accel_pub.publish(self.accel_cmd)
                        self.brake_cmd.f64_cmd = 1.0
                        self.brake_pub.publish(self.brake_cmd)
                else:
                    self.maintain_speed()
                    self.make_quarter_circle(-self.steering_angle, RIGHT)

            else:
                print("no centroids found yet")
                self.stop()
            self.rate.sleep()


    def heading_to_yaw(self, heading_curr):
        # 0   <= heading < 90  --- 90 to 0     (pi/2 to 0)
        # 90  <= heading < 180 --- 0 to -90    (0 to -pi/2)
        # 180 <= heading < 270 --- -90 to -180 (-pi/2 to -pi)
        # 270 <= heading < 360 --- 180 to 90   (pi to pi/2)
        if (heading_curr >= 0 and heading_curr < 90):
            yaw_curr = np.radians(90 - heading_curr)
        elif(heading_curr >= 90 and heading_curr < 180):
            yaw_curr = np.radians(90 - heading_curr)
        elif(heading_curr >= 180 and heading_curr < 270):
            yaw_curr = np.radians(90 - heading_curr)
        else:
            yaw_curr = np.radians(450 - heading_curr)
        return yaw_curr


# computes the Euclidean distance between two 2D points
def dist(p1, p2):
    return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)


def circle_box():

    rospy.init_node('circle_box', anonymous=True)
    cb = CircleBox()

    try:
        cb.start_circlebox()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    circle_box()