#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import numpy as np
import socket
import rosbag

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes

def callback(boxes):
    print(boxes)
    pass

def main():
    rospy.init_node('output_receiver', anonymous=True)
    recv_topic = rospy.get_param(
            '~recv_topic', '/yolov5/BoundingBoxes')
    
    sub = rospy.Subscriber(recv_topic, BoundingBoxes, callback,
                                          queue_size=1, buff_size=52428800)
    rospy.spin()

if __name__ == "__main__":
    main()