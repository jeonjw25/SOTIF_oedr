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

global cnt1
global cnt2

cnt1 = 0
cnt2 = 0

class Output_Recv:
    def __init__(self):
        box_topic = rospy.get_param(
                '~box_topic', '/yolov5/BoundingBoxes')
        l_image_topic = rospy.get_param(
                '~l_image_pub' , '/yolov5/detection_image_left')  
        r_image_topic = rospy.get_param(
                '~r_image_pub' , '/yolov5/detection_image_right')

        self.bboxes = None
        self.l_img = None
        self.r_img = None

        self.sub_box = rospy.Subscriber(box_topic, BoundingBoxes, self.box_callback,
                                            queue_size=1, buff_size=52428800)
        self.sub_l_image = rospy.Subscriber(l_image_topic, Image, self.left_img_callback,
                                            queue_size=1, buff_size=52428800)
        self.sub_r_image = rospy.Subscriber(r_image_topic, Image, self.right_img_callback,
                                            queue_size=1, buff_size=52428800)                                    
        
        cx = 959.724
        cy = 545.2921
        fx = 2065
        fy = 2065

        self.Q = np.array([[1, 0, 0, -cx],
                      [0, 1, 0, -cy],
                      [0, 0, 0, fx],
                      [0, 0, -1, 0]])

        # cv2.setUseOptimized(True)
        # cv2.setNumThreads(4)
        # cv2.ocl.setUseOpenCL(True)
        

    def box_callback(self, boxes):
        print(boxes)
        self.bboxes = boxes.bounding_boxes
        # print(self.bboxes)
        
    def left_img_callback(self, img):
        self.l_img = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
        # self.l_img = np.array(img.data, dtype=np.uint8).reshape(img.height, img.width)

        # self.l_img = cv2.cvtColor(self.l_img, cv2.COLOR_BGR2RGB)

    def right_img_callback(self, img):
        self.r_img = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
        # self.r_img = np.array(img.data, dtype=np.uint8).reshape(img.height, img.width)
        # self.r_img = cv2.cvtColor(self.r_img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('rrr', self.r_img)
        # cv2.waitKey(3)
        distance = self.cal_distance()
        #print(cv2.cuda.getCudaEnabledDeviceCount())
        #print(cv2.cuda.DeviceInfo())
        # print(distance)
        

    def cal_distance(self):
        bboxes = self.bboxes
        l_img = cv2.cvtColor(self.l_img, cv2.COLOR_RGB2GRAY)
        r_img = cv2.cvtColor(self.r_img, cv2.COLOR_RGB2GRAY)
        # l_img = self.l_img
        # r_img = self.r_img
        
        width = l_img.shape[1]
        distances = []
        
        for box in bboxes:
           class_list = ['truck', 'car', 'bus']
           if box.Class in class_list:
                x_min = box.xmin
                x_max = box.xmax
                y_min = box.ymin
                y_max = box.ymax

                
                stream = cv2.cuda_Stream()
                left_cuda = cv2.cuda_GpuMat(l_img)
                right_cuda = cv2.cuda_GpuMat(r_img)
                # stereo = cv2.StereoSGBM_create(numDisparities=128, blockSize=19, speckleWindowSize=10)
                # disparity = stereo.compute(l_img,r_img)

                stereo = cv2.cuda.createStereoSGM(minDisparity = 0, numDisparities=64, uniquenessRatio=25)
                # stereo = cv2.cuda.createStereoBM(numDisparities=128, blockSize=19)
               
                disparity_cuda = stereo.compute(right_cuda, left_cuda)
                disparity = disparity_cuda.download()

                # points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
                # print(points_3d.shape)

                focal_length = 2065 # need to transfer to m
                baseline = 1 # m
                depth_map = baseline * focal_length / disparity
                cv2.imshow('depth map', disparity)
                global cnt1
                
                cv2.waitKey(3)
                # nDisparity = np.array(disparity)+17
                # nDisparity = 29184/(nDisparity*2)
                cv2.imwrite('/root/catkin_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_ros/media/depth'+str(cnt1)+'.png', disparity)
                cnt1 += 1
                x_left = max(0,x_min)
                x_right = min(width, x_max)
                depth_map = np.array(depth_map)
                distance = np.round(np.amin(depth_map[y_min:y_max, x_left:x_right]), 2)
                # distance = np.round(np.amin(nDisparity[y_min:y_max, x_min:x_max]), 2)
                if y_min < 20:
                    text_pos_y = y_min + 30
                else:
                    text_pos_y = y_min - 10

                cv2.putText(self.r_img, str(distance)+'m',
                            (int(x_min), int(text_pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                distances.append(distance)
        
        # cv2.imshow('add depth', self.r_img)
        global cnt2
        # cv2.imwrite('/root/catkin_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_ros/media/distance'+str(cnt2)+'.png', self.r_img)
        cnt2 += 1
        cv2.waitKey(3)
        return distances

def main():
    rospy.init_node('output_receiver', anonymous=True)
    output_recv = Output_Recv()
    rospy.spin()
    

if __name__ == "__main__":
    main()
