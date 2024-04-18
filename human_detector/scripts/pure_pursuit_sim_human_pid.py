#!/usr/bin/env python3

#================================================================
# File name: pure_pursuit_sim.py                                                                  
# Description: pure pursuit controller for GEM vehicle in Gazebo                                                              
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 07/10/2021                                                                
# Date last modified: 07/15/2021                                                          
# Version: 0.1                                                                    
# Usage: rosrun gem_pure_pursuit_sim pure_pursuit_sim.py                                                                    
# Python version: 3.8                                                             
#================================================================

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import torch
import cv2
from PIL import Image as im
import time
import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ROS Headers
import rospy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import Twist, Vector3
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Gazebo Headers
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState

# PID controller
from simple_pid.pid import PID


SHIFT_REVERSE_VALUE = 1
SHIFT_NEUTRAL_VALUE = 2
SHIFT_FORWARD_VALUE = 3

DISENGAGE_MESSAGE_VALUE = 0.0
BRAKE_MESSAGE_VALUE = 0.5
# ACCELERATE_MESSAGE_VALUE = 0.4

# prevent quick switching between forward/reverse
# will the lack of motion cause the pid controller to output a bigger acc value?
# is it better to artifically set the pid input equal to setpoint when similar?
# or do both ?
MOTION_MARGIN = 0.2

PID_SETPOINT = 60000 # Area of bounding box # idk actual reasonable value yet
PID_SETPOINT_MARGIN = 5000
ACC_LIMIT_LOW = -0.5
ACC_LIMIT_HIGH = 0.5

HUMAN_CLASS = 0

class PurePursuit(object):
    
    def __init__(self):

        self.rate       = rospy.Rate(20)

        self.look_ahead = 6    # meters
        self.wheelbase  = 1.75 # meters
        self.goal       = 0

        self.read_waypoints() # read waypoints

        self.ackermann_msg = AckermannDrive()
        self.ackermann_msg.steering_angle_velocity = 0.0
        self.ackermann_msg.acceleration            = 0.0
        self.ackermann_msg.jerk                    = 0.0
        self.ackermann_msg.speed                   = 0.0 
        self.ackermann_msg.steering_angle          = 0.0

        self.human_detected = False
        self.bb_size = None

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device=0)

        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.detect_human)

        # self.image_pub = rospy.Publisher("/bounding_img", Image, queue_size=15)

        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)
        self.log = open('pid_log.txt', 'a')

        kp, ki, kd = 1, 0.1, 0.05 # update ?
        self.pid = PID(kp, ki, kd, setpoint=PID_SETPOINT,output_limits = (ACC_LIMIT_LOW, ACC_LIMIT_HIGH))




    # import waypoints.csv into a list (path_points)
    def read_waypoints(self):

        dirname  = os.path.dirname(__file__)
        print('\n\n' + os.path.abspath(dirname) + '\n\n')
        filename = os.path.join(dirname, '../../POLARIS_GEM_e2_Simulator/vehicle_drivers/gem_pure_pursuit_sim/waypoints/wps.csv')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]

        # turn path_points into a list of floats to eliminate the need for casts
        self.path_points_x   = [float(point[0]) for point in path_points]
        self.path_points_y   = [float(point[1]) for point in path_points]
        self.path_points_yaw = [float(point[2]) for point in path_points]
        self.dist_arr        = np.zeros(len(self.path_points_x))

    # computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    # find the angle bewtween two vectors    
    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        # [-pi, pi]
        return np.arctan2(sinang, cosang)

    def get_gem_pose(self):

        rospy.wait_for_service('/gazebo/get_model_state')
        
        try:
            service_response = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            model_state = service_response(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: " + str(exc))

        x = model_state.pose.position.x
        y = model_state.pose.position.y

        orientation_q      = model_state.pose.orientation
        orientation_list   = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        return round(x,4), round(y,4), round(yaw,4)


    def start_pp(self):
        
        while not rospy.is_shutdown():

            # get current position and orientation in the world frame
            curr_x, curr_y, curr_yaw = self.get_gem_pose()

            self.path_points_x = np.array(self.path_points_x)
            self.path_points_y = np.array(self.path_points_y)

            # finding the distance of each way point from the current position
            for i in range(len(self.path_points_x)):
                self.dist_arr[i] = self.dist((self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y))

            # finding those points which are less than the look ahead distance (will be behind and ahead of the vehicle)
            goal_arr = np.where( (self.dist_arr < self.look_ahead + 0.3) & (self.dist_arr > self.look_ahead - 0.3) )[0]

            # finding the goal point which is the last in the set of points less than the lookahead distance
            for idx in goal_arr:
                v1 = [self.path_points_x[idx]-curr_x , self.path_points_y[idx]-curr_y]
                v2 = [np.cos(curr_yaw), np.sin(curr_yaw)]
                temp_angle = self.find_angle(v1,v2)
                if abs(temp_angle) < np.pi/2:
                    self.goal = idx
                    break

            # finding the distance between the goal point and the vehicle
            # true look-ahead distance between a waypoint and current position
            L = self.dist_arr[self.goal]

            # transforming the goal point into the vehicle coordinate frame 
            gvcx = self.path_points_x[self.goal] - curr_x
            gvcy = self.path_points_y[self.goal] - curr_y
            goal_x_veh_coord = gvcx*np.cos(curr_yaw) + gvcy*np.sin(curr_yaw)
            goal_y_veh_coord = gvcy*np.cos(curr_yaw) - gvcx*np.sin(curr_yaw)

            # find the curvature and the angle 
            alpha   = self.path_points_yaw[self.goal] - (curr_yaw)
            k       = 0.285
            angle_i = math.atan((2 * k * self.wheelbase * math.sin(alpha)) / L) 
            angle   = angle_i*2
            angle   = round(np.clip(angle, -0.61, 0.61), 3)

            ct_error = round(np.sin(alpha) * L, 3)

            print("Crosstrack Error: " + str(ct_error))

            # implement constant pure pursuit controller
            # self.ackermann_msg.speed          = 2.8
            # self.ackermann_msg.steering_angle = angle
            # self.ackermann_pub.publish(self.ackermann_msg)

            if self.human_detected:
                # self.ackermann_msg.speed          = 0.0
                # self.ackermann_pub.publish(self.ackermann_msg)
                # get pid control from detector box size
                # add margin to prevent rapidly switching forward/reverse
                self.log.write('%s: Setpoint: %f, BB Size: %f \n'% (datetime.datetime.now(), PID_SETPOINT, self.bb_size))
                if abs(self.bb_size - PID_SETPOINT) < PID_SETPOINT_MARGIN:
                    control = self.pid(PID_SETPOINT)
                else:
                    control = self.pid(self.bb_size)

                control = self.pid(self.bb_size)
                print("PID controller output:", control)
                self.log.write('PID Control: %f\n---\n'% (control))

                if control > MOTION_MARGIN or control < -MOTION_MARGIN:
                    self.ackermann_msg.speed = control
                    self.ackermann_msg.steering_angle = angle
                    self.ackermann_pub.publish(self.ackermann_msg)
                    print("Controlling speed to human")
                else:
                    self.ackermann_msg.speed = 0.0
                    self.ackermann_msg.steering_angle = angle
                    self.ackermann_pub.publish(self.ackermann_msg)
                    print("Human within distance margin")
            else:
                self.ackermann_msg.speed = 0.0
                self.ackermann_msg.steering_angle = angle
                self.ackermann_pub.publish(self.ackermann_msg)
                print("No human detected")


            self.rate.sleep()
            # rospy.spin()

    def detect_human(self, image):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(image, "bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.model(img)
        results.print()
        results.render()
        
        box_df = results.pandas().xyxy[0]
        people_df = box_df.loc[box_df['class'] == 0]
        print(people_df.size)
        for index, row in people_df.iterrows():
            print("Size: ", (row['xmax']-row['xmin']) * (row['ymax']-row['ymin']))
    
        self.human_detected = people_df.size > 0
        if self.human_detected:
            person = people_df.loc[0]
            self.bb_size = (person['xmax'] - person['xmin']) * (person['ymax'] - person['ymin'])
        else:
            self.bb_size = None
    

def pure_pursuit():

    rospy.init_node('pure_pursuit_sim_human_pid', anonymous=True)
    pp = PurePursuit()

    try:
        pp.start_pp()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    pure_pursuit()

