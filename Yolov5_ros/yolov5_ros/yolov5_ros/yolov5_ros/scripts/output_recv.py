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

global cnt
cnt = 0

class Output_Recv:
    def __init__(self):
        box_topic = rospy.get_param(
                '~box_topic', '/yolov5/BoundingBoxes')
        l_image_topic = rospy.get_param(
                '~l_image_pub' , '/yolov5/detection_image_left')  
        r_image_topic = rospy.get_param(
                '~r_image_pub' , '/yolov5/detection_image_right')

        self.sub_box = rospy.Subscriber(box_topic, BoundingBoxes, self.box_callback,
                                            queue_size=1, buff_size=52428800)
        self.sub_l_image = rospy.Subscriber(l_image_topic, Image, self.left_img_callback,
                                            queue_size=1, buff_size=52428800)
        self.sub_r_image = rospy.Subscriber(r_image_topic, Image, self.right_img_callback,
                                            queue_size=1, buff_size=52428800)                                    
        
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)
        cv2.ocl.setUseOpenCL(True)
        self.bboxes = None
        self.l_img = None
        self.r_img = None

    def box_callback(self, boxes):
        self.bboxes = boxes.bounding_boxes

        
    def left_img_callback(self, img):
        self.l_img = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
        # self.l_img = cv2.cvtColor(self.l_img, cv2.COLOR_BGR2RGB)

    def right_img_callback(self, img):
        self.r_img = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
        # self.r_img = cv2.cvtColor(self.r_img, cv2.COLOR_BGR2RGB)
        
        distance = self.cal_distance()
        print(cv2.cuda.getCudaEnabledDeviceCount())
        print(distance)
        

    def cal_distance(self):
        bboxes = self.bboxes
        l_img = cv2.cvtColor(self.l_img, cv2.COLOR_BGR2GRAY)
        r_img = cv2.cvtColor(self.r_img, cv2.COLOR_BGR2GRAY)
        width = l_img.shape[1]
        distances = []
        
        for box in bboxes:
           class_list = ['truck', 'car', 'bus']
           if box.Class in class_list:
                x_min = box.xmin
                x_max = box.xmax
                y_min = box.ymin
                y_max = box.ymax

                stereo = cv2.StereoBM_create(numDisparities=256, blockSize=19)
                disparity = stereo.compute(r_img,l_img)
                cv2.imshow('depth map', disparity)
                cv2.waitKey(3)
                nDisparity = np.array(disparity)+17
                nDisparity = 29184/(nDisparity*2)
                x_left = max(0,x_min-(x_max-x_min))
                x_right = min(width, x_max+(x_max-x_min))
                distance = np.round(np.amin(nDisparity[y_min:y_max, x_min:x_max]), 2)
                cv2.putText(self.r_img, str(distance)+'m',
                            (x_min+5, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                distances.append(distance)
        
        cv2.imshow('add depth', self.r_img)
        global cnt
        cv2.imwrite('/root/catkin_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_ros/media/depth'+str(cnt)+'.png', self.r_img)
        cnt += 1
        cv2.waitKey(3)
        return distances

def main():
    rospy.init_node('output_receiver', anonymous=True)
    output_recv = Output_Recv()
    rospy.spin()
    

if __name__ == "__main__":
    main()