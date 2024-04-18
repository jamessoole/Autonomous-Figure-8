#!/usr/bin/env python3

#================================================================
# File name: gem_gnss_pp_tracker_pid.py                                                                  
# Description: gnss waypoints tracker using pid and pure pursuit                                                                
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 08/02/2021                                                                
# Date last modified: 08/13/2021                                                          
# Version: 0.1                                                                    
# Usage: rosrun gem_gnss gem_gnss_pp_tracker.py                                                                      
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

# ROS Headers
import alvinxy.alvinxy as axy # Import AlvinXY transformation module
import rospy

from nav_msgs.msg import Path

from geometry_msgs.msg import PoseStamped, PoseArray, Pose

# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd
from tf.transformations import euler_from_quaternion, quaternion_from_euler


import matplotlib.pyplot as plt

#import cvxpy

# ROS Headers
from PythonRobotics.PathPlanning.DubinsPath import dubins_path_planner


NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 5  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 1.75  # [m]

MAX_STEER = 0.61  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 2.8  # maximum speed [m/s]
MIN_SPEED = 1.0  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]


LEFT_BOX = 0
RIGHT_BOX = 1


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


class PurePursuit(object):
    
    def __init__(self):

        self.rate       = rospy.Rate(10)
        # self.log = open('log3.txt', 'a')

        # self.look_ahead = 4

        # threshold for when to move to next waypoint (distance in meters)
        self.delta = 1.5

        self.done = False

        self.curvature = 0.3
        self.step_size = 0.3

        self.wheelbase  = 1.75 # meters
        self.offset     = 0.46 # meters

        self.gnss_sub   = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        self.lat        = 0.0
        self.lon        = 0.0
        self.heading    = 0.0
        self.first = True

        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

        self.speed_sub  = rospy.Subscriber("/pacmod/as_tx/vehicle_speed", Float64, self.speed_callback)
        self.speed      = 0.0

        self.path = None
        self.path_pub = rospy.Publisher('/path', Path, queue_size=1)

        self.olat       = 40.0928563 # original lat/long # TODO, need to find where this is physically and start car here
        self.olon       = -88.2359994
        # self.olat       = 40.092855 # original lat/long # TODO, need to find where this is physically and start car here
        # self.olon       = -88.235838

        # x,y,yaw = self.get_gem_state()
        # self.olat += x
        # self.olon += y

        # read waypoints into the system 
        # self.prev_goal = -1
        self.goal       = 0            
        # self.read_waypoints() 
        
        self.box_direction = LEFT_BOX  # todo: change this programatically

        self.desired_speed = 0.5  # m/s, reference speed
        self.max_accel     = 0.4 # % of acceleration
        self.pid_speed     = PID(1.5, 0.3, 0.6, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

        # -------------------- PACMod setup --------------------

        self.gem_enable    = False
        self.pacmod_enable = False

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
        self.steer_cmd.angular_velocity_limit = 2.0 # radians/second

        # midpoint callback
        self.midpoint_sub = rospy.Subscriber("midpoint", PoseArray, self.midpoint_callback)

        self.midpoint = []
        self.box_centroids = (None, None)


    # gets the midpoint from the lidar node
    def midpoint_callback(self, msg):
        if self.first:
            self.midpoint = [msg.poses[2].position.x, msg.poses[2].position.y, 0]
            self.box_centroids = ((msg.poses[0].position.x, msg.poses[0].position.y, 0),
                                (msg.poses[1].position.x, msg.poses[1].position.y, 0))
        print("Midpoint: ", self.midpoint)
        print("Box0: ", self.box_centroids[0])
        print("Box1: ", self.box_centroids[1])


    def inspva_callback(self, inspva_msg):
        self.lat     = inspva_msg.latitude  # latitude
        self.lon     = inspva_msg.longitude # longitude
        self.heading = inspva_msg.azimuth   # heading in degrees # from where? x-axis(east)?
        # x,y,h = self.get_gem_state()
        # print("x,y,h: %f,%f,%f\n"% (self.lat,self.lon,self.heading))
        # self.log.write('%f,%f,%f\n'% (x,y,self.heading))
        # temp_lat = self.olat-self.lat
        # temp_lon = self.olon-self.lon
        # self.log.write('%f,%f,%f\n'%(x, y, self.heading))

    def speed_callback(self, msg):
        self.speed = round(msg.data, 3) # forward velocity in m/s

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

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

    def front2steer(self, f_angle):

        if(f_angle > 35):
            f_angle = 35

        if (f_angle < -35):
            f_angle = -35

        return f_angle

        # if (f_angle > 0):
        #     steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)

        # elif (f_angle < 0):
        #     f_angle = -f_angle
        #     steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        # else:
        #     steer_angle = 0.0

        # return steer_angle

    # def read_waypoints(self):
    #     # read recorded GPS lat, lon, heading
    #     dirname  = os.path.dirname(__file__)
    #     filename = os.path.join(dirname, '../waypoints/log3.csv')
    #     # filename = os.path.join(dirname, '../waypoints/xy_demo.csv') # loop
    #     # filename = os.path.join(dirname, '../waypoints/xy_demo_1.csv') # wiggly loop

    #     with open(filename) as f:
    #         path_points = [tuple(line) for line in csv.reader(f)]

    #     # x towards East and y towards North
    #     self.path_points_lon_x   = [float(point[0]) for point in path_points] # longitude
    #     self.path_points_lat_y   = [float(point[1]) for point in path_points] # latitude
    #     self.path_points_heading = [float(point[2]) for point in path_points] # heading
    #     self.wp_size             = len(self.path_points_lon_x)
    #     self.dist_arr            = np.zeros(self.wp_size)

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

    # computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)
    
    def get_angle_between_points(self, p1, p2):
      return np.arctan2((p1[1] - p2[1]),(p1[0] - p2[0]))


    def distance(self, p1, p2):
      return ((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)**0.5
    
    def follow_waypoints(self):
      # compute current waypoint x and y and current car x and y
      wp_x = self.path_points_lon_x[self.goal]
      wp_y = self.path_points_lat_y[self.goal]
      wp_heading = self.path_points_heading[self.goal]

      curr_x, curr_y, curr_yaw = self.get_gem_state()


      print("Current x, y, yaw: " + str([curr_x, curr_y, curr_yaw]))
      print("Current waypoint x, y, yaw: " + str([wp_x, wp_y, wp_heading]))
      print("midpoint: ", self.midpoint[0], self.midpoint[1])

      # find the curvature and the angle 
      alpha = self.heading_to_yaw(wp_heading) - curr_yaw

      # compute distance between current location and current waypoint
      L = self.distance((wp_x, wp_y), (curr_x, curr_y))
      print("L: " + str(L))
      # set self.goal to "next" waypoint if distance to waypoint < delta
      if L < self.delta:
          self.goal += 1

          print("num waypoints: " + str(len(self.path_points_lon_x)))
          if self.goal >= len(self.path_points_lon_x) - 1:
            print("BRAKING")
            self.accel_cmd.f64_cmd = 0
            self.accel_pub.publish(self.accel_cmd)
            self.brake_cmd.f64_cmd = 1.0
            self.brake_pub.publish(self.brake_cmd)
            self.done = True
            return

          ### uncomment this to automatically append circle
          #   if self.goal >= len(self.wp_size):
          #   self.append_circle_to_waypoint(self.box_direction)


      

      # ---------- PID things - tuning this part as needed -----------------
      k       = 0.41 
      angle_i = math.atan((k * 2 * self.wheelbase * math.sin(alpha)) / L) 
      angle   = angle_i*2
      # ----------------- tuning this part as needed -----------------

      f_delta = round(np.clip(angle, -0.61, 0.61), 3)
      f_delta_deg = np.degrees(f_delta)

      # steering_angle in degrees
      steering_angle = self.front2steer(f_delta_deg)
    

      # print debug info
      if(self.gem_enable == True):
          print("Waypoint size: " + str(len(self.path_points_lat_y)))
          print("Current goal index: " + str(self.goal))
          print("Forward velocity: " + str(self.speed))
          ct_error = round(np.sin(alpha) * L, 3)
          print("Crosstrack Error: " + str(ct_error))
          print("Front steering angle: " + str(np.degrees(f_delta)) + " degrees")
          print("Steering wheel angle: " + str(steering_angle) + " degrees" )
          print("\n")
      else:
          print('self.gem_enable NOT ENABLED')

      if self.path is not None: 
          self.path_pub.publish(self.path)

      # controls stuff
      current_time = rospy.get_time()
      filt_vel     = self.speed_filter.get_data(self.speed)
      output_accel = self.pid_speed.get_control(current_time, self.desired_speed - filt_vel)

      if output_accel > self.max_accel:
          output_accel = self.max_accel

      if output_accel < 0.3:
          output_accel = 0.3

      if (f_delta_deg <= 30 and f_delta_deg >= -30):
          self.turn_cmd.ui16_cmd = 1
      elif(f_delta_deg > 30):
          self.turn_cmd.ui16_cmd = 2 # turn left
      else:
          self.turn_cmd.ui16_cmd = 0 # turn right

      self.accel_cmd.f64_cmd = output_accel
      self.steer_cmd.angular_position = np.radians(steering_angle)
      self.accel_pub.publish(self.accel_cmd)
      self.steer_pub.publish(self.steer_cmd)
      self.turn_pub.publish(self.turn_cmd)


    def append_circle_to_waypoint(self, box=LEFT_BOX):
        if box == LEFT_BOX:
            self.path_points_lon_x = np.concatenate((self.path_points_lon_x, self.left_circle_xs))
            self.path_points_lat_y = np.concatenate((self.path_points_lat_y, self.left_circle_ys))
            self.path_points_heading = np.concatenate((self.path_points_heading, self.left_circle_yaws))
        else:
            self.path_points_lon_x = np.concatenate((self.path_points_lon_x, self.right_circle_xs))
            self.path_points_lat_y = np.concatenate((self.path_points_lat_y, self.right_circle_ys))
            self.path_points_heading = np.concatenate((self.path_points_heading, self.right_circle_yaws))


    def get_circle_points(self, box_1_coordinates, box_2_coordinates, center_coordinates, car_heading, box=LEFT_BOX, num_points=10):
      unit_vec = np.array([np.cos(car_heading), np.sin(car_heading)])
      box_1_vec = np.array(box_1_coordinates) - np.array(center_coordinates)
      
      cross_product = np.cross(unit_vec, box_1_vec)
      if cross_product > 0:
        left_box_coordinates, right_box_coordinates = box_1_coordinates, box_2_coordinates 
      else:
        left_box_coordinates, right_box_coordinates = box_2_coordinates, box_1_coordinates 

      # determine which box we are circling
      box_coordinates = left_box_coordinates if box==LEFT_BOX else right_box_coordinates

      radius = self.distance(box_coordinates,center_coordinates)

      angle_between_box_and_car = self.get_angle_between_points(box_coordinates, center_coordinates)

      angles = np.array([angle_between_box_and_car + 2 * np.pi / num_points * i for i in range(1,num_points+1)])
      headings = np.array([angle + np.pi / 2 for angle in angles])

      wps = np.array([np.array([box_coordinates[0] + np.cos(angle) * radius, box_coordinates[1] + np.sin(angle) * radius]) for angle in angles])

      xs = wps[:,0]
      ys = wps[:,1]

      if box == RIGHT_BOX:
        xs = np.roll(np.flip(xs), -1)
        ys = np.roll(np.flip(ys), -1)
        headings = np.roll(np.flip(headings) + np.pi, -1)
        
      print("X coordinates:")
      print(xs)
      print("Y coordinates:")
      print(ys)
      print("Headings:")
      print(headings)

      return xs, ys, headings

    def start_pp(self):
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

            # get current position and orientation in the world frame
            curr_x, curr_y, curr_yaw = self.get_gem_state()
            print(curr_x, curr_y, curr_yaw)

            if self.midpoint is not None and self.box_centroids[0] is not None and self.box_centroids[1] is not None and self.first:   
              y_diff = self.box_centroids[1][1]-self.box_centroids[0][1]
              x_diff = self.box_centroids[1][0]-self.box_centroids[0][0]

              midpoint_angle = 0
              if x_diff == 0 and y_diff > 0:
                  midpoint_angle = np.pi/2
              elif x_diff == 0 and y_diff < 0:
                  midpoint_angle = -np.pi/2
              else:
                  print("using arctan to calculate midpoint heading")
                  midpoint_angle = np.arctan2(y_diff, x_diff)
              midpoint_angle1 = midpoint_angle + (np.pi/2)
              midpoint_angle2 = midpoint_angle - (np.pi/2)

              print("midpoint angle 1: " + str(midpoint_angle1))
              print("midpoint angle 2: " + str(midpoint_angle2))

              step1 = (self.midpoint[0] + np.cos(midpoint_angle1), self.midpoint[1] + np.sin(midpoint_angle1))
              step2 = (self.midpoint[0] + np.cos(midpoint_angle2), self.midpoint[1] + np.sin(midpoint_angle2))

              print("step1: " + str(step1))
              print("step2: " + str(step2))

              distance_1 = ((curr_x - step1[0])**2 + (curr_y - step1[1])**2)**0.5
              distance_2 = ((curr_x - step2[0])**2 + (curr_y - step2[1])**2)**0.5

              print("distance_1: " + str(distance_1))
              print("distance_2: " + str(distance_2))

              # todo: check this, but it makes sense. we want our heading to be opposite the smaller distance
              theta = midpoint_angle1 if distance_1 > distance_2 else midpoint_angle2
              self.midpoint[2] = theta
              
              print("theta: " + str(theta))
              
              self.path_points_lon_x, self.path_points_lat_y, self.path_points_heading, _,_ = dubins_path_planner.plan_dubins_path(0, 0, curr_yaw - (np.pi/2),
                                                                  self.midpoint[0], self.midpoint[1], theta,
                                                                  self.curvature, self.step_size)
              
              # calculate path
              self.path = Path()
              self.path.header.frame_id = "base_footprint"
              for i in range(len(self.path_points_lon_x)):
                    mp = PoseStamped()
                    mp.header.stamp = rospy.Time(i)
                    mp.header.frame_id = "base_footprint"
                    mp.pose.position.x = self.path_points_lon_x[i]
                    mp.pose.position.y = self.path_points_lat_y[i]
                    ori = quaternion_from_euler(0.0, 0.0, self.path_points_heading[i])
                    mp.pose.orientation.x = ori[0]
                    mp.pose.orientation.y = ori[1]
                    mp.pose.orientation.z = ori[2]
                    mp.pose.orientation.w = ori[3]
                    self.path.poses.append(mp)
              
              self.path_points_lon_x, self.path_points_lat_y, self.path_points_heading = np.array(self.path_points_lon_x), np.array(self.path_points_lat_y), np.array(self.path_points_heading)

              self.path_points_lon_x += curr_x
              self.path_points_lat_y += curr_y

              self.path_points_heading += curr_yaw

              # trim path points

              # calculate circle coordinates
              self.left_circle_xs, self.left_circle_ys, self.left_circle_yaws = self.get_circle_points(self.box_centroids[0][0:2], self.box_centroids[1][0:2], self.midpoint[0:2], self.midpoint[2], LEFT_BOX, 10)

              self.right_circle_xs, self.right_circle_ys, self.right_circle_yaws = self.get_circle_points(self.box_centroids[0][0:2], self.box_centroids[1][0:2], self.midpoint[0:2], self.midpoint[2], RIGHT_BOX, 10)

              # append left circle coordinates
              # append_circle_to_waypoint(LEFT_BOX)

              self.wp_size             = len(self.path_points_lon_x)
              self.dist_arr            = np.zeros(self.wp_size)
              self.first = False

            if not self.done and not self.first and self.midpoint is not None and self.path_points_lon_x is not None and len(self.path_points_lon_x) != 0 and not self.first:
              self.follow_waypoints()

            if self.done:
                print("done!")

            self.rate.sleep()


def pure_pursuit():

    rospy.init_node('gnss_pp_node', anonymous=True)
    pp = PurePursuit()

    try:
        pp.start_pp()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pure_pursuit()