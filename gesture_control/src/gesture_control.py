#!/usr/bin/python3

from __future__ import print_function

# Python imports
import math
import numpy as np
import cv2

import torch

# ROS imports
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

CAMERA_TOPIC = '/zed2/zed_node/rgb_raw/image_raw_color'
NEW_CAM_TOPIC = '/new_img'

HUMAN_CLASS = 0

### Gesture control map
# Maps gesture prediciton strings to actions
# TODO: map to waypoints? or turn commands? figure out later
gesture_map = {
    "Backward": "Backward",
    "Forward": "Forward",
    "Go": "Go",
    "None": "None",
    "Stop": "Stop",
    "Turn Left": "Turn Left",
    "Turn Right": "Turn Right"
}

class GestureControl(object):
    def __init__(self):
        
        self.gesture_subscriber = rospy.Subscriber("/gesture/hand_sign", String, self.gesture_callback)
        self.gesture = gesture_map["None"]

        # Load the YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device="cpu")
        self.human_detected = False
        self.bb_size = None

         # Subscribe to camera topic
        self.image_sub = rospy.Subscriber(CAMERA_TOPIC, Image, self.detect_human)

        self.new_image_pub = rospy.Publisher(NEW_CAM_TOPIC, Image, queue_size=10)

        self.current_cmd = "Stop"


    def run(self):
        while not rospy.is_shutdown():
            if self.current_cmd is "Stop":
                self.current_cmd = self.gesture

            self.execute_gesture()
            # print("Gesture: ", self.gesture)
            # continue
            rospy.spin()
        
    def gesture_callback(self, str_msg):
        """A callback function for the gesture label
        Sets the value of self.gesture to whatever 
        prediction the gesture recognition model
        makes.
        
        Args:
            str_msg (std_msgs.msg): string message
        """
        if str_msg.data is not "None":
            self.gesture = gesture_map[str_msg.data]

        if self.gesture == "Stop":
            # self.stop()
            print("STOP")
            self.current_cmd = "Stop"

    def execute_gesture(self):
        # self.maintain_speed()
        print(gesture_map[self.current_cmd])
        return

    def detect_human(self, image):
        # convert image for model
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(image, "bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get model results
        results = self.model(img)
        results.print()
        # results.render()
        # convert to pandas df
        
        crops = results.crop(save=False)
        c = []
        for i in range(len(crops)):
            crop = crops[i]
            if str(crop['label']).startswith('person'):
                c.append(crop)
                # print(len(crops))
        box_df = results.pandas().xyxy[0]
        # print(box_df)

        # get result with humans detected
        people_df = box_df.loc[box_df['class'] == HUMAN_CLASS]
        # ind = box_df.index[box_df['class'] == HUMAN_CLASS].tolist()
        self.human_detected = people_df.size > 0
        # print(people_df)

        # get bounding box size for one person detected
        # (could/should also do biggest/closest person)
        if self.human_detected:
            # c = crops[ind[0]]
            person = people_df.iloc[0]
            self.bb_size = (person['ymax'] - person['ymin'])
            new_img = c[0]['im']
            dim = (int(img.shape[1] * 2), int(img.shape[0] * 2))
            # print(new_img.shape)
            new_img = cv2.resize(new_img, dim, interpolation=cv2.INTER_CUBIC)
            # print(new_img.shape)
            # img_msg = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
            img_msg = bridge.cv2_to_imgmsg(new_img)
            self.new_image_pub.publish(img_msg)
            print('self.bb_size:', self.bb_size)
        else:
            print('No Person')
            self.bb_size = None

def gesture_control():
    rospy.init_node('gesture-control', anonymous=True)
    gc = GestureControl()

    try:
        gc.run()
    except rospy.ROSInterruptException:
        pass
    
if __name__ == '__main__':
    gesture_control()
        