#!/usr/bin/env python3
from __future__ import print_function

#Python Headers
import math
import os
import torch
import cv2
import numpy as np
from PIL import Image as im
import time
import datetime

# ROS Headers
import rospy
import roslaunch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# GEM PACMod Headers
from std_msgs.msg import Header
# from pacmod_msgs.msg import PacmodCmd, PositionWithSpeed, VehicleSpeedRpt

class Node():

  def __init__(self):
    self.rate = rospy.Rate(10)

    # Load the YOLO model
    self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device=0)

    self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.process_image)

    self.image_pub = rospy.Publisher("/bounding_img", Image, queue_size=15)
    self.log = open('log.txt', 'a')

  def process_image(self, msg):
    start = time.perf_counter()
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    img0 = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = self.model(img)
    results.print()
    results.render()
    
    box_df = results.pandas().xyxy[0]
    people_df = box_df.loc[box_df['class'] == 0]
    for index, row in people_df.iterrows():
      msg = "Size: ", (row['xmax']-row['xmin']) * (row['ymax']-row['ymin'])
      self.log.write('%s: %s\n---\n'% (datetime.datetime.now(), msg))
    
    
    img_msg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_msg = bridge.cv2_to_imgmsg(img, encoding="rgb8")
    self.image_pub.publish(img_msg)
    print("\nTime: ", time.perf_counter() - start)
    

  def run(self):
    while not rospy.is_shutdown():

      self.rate.sleep
      rospy.spin()

if __name__ == '__main__':
	rospy.init_node('human_detect_node', anonymous=True)
	node = Node()
	node.run()
