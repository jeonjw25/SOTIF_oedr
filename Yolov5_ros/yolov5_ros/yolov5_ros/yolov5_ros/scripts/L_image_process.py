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
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from oedr_utils import calculate_curvature, crop

global cnt
cnt = 0

class Yolo_Dect:
    def __init__(self):

        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '')

        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param(
            '~image_topic_l', '/vds_node_localhost_2210/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')

        # load local repository(YoloV5:v6.0)
        self.model = torch.hub.load(yolov5_path, 'custom',
                                    path=weight_path, source='local')

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.conf = conf
        self.color_image = Image()
        self.depth_image = Image()
        self.getImageStatus = False
        
        # Load class color
        self.classes_colors = {}

        # image subscribe
        # print(image_topic)
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                          queue_size=1, buff_size=52428800)                               

        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=1)

        self.l_image_pub = rospy.Publisher(
            '/yolov5/detection_image_left',  Image, queue_size=1)

        # if no image messages
        while (not self.getImageStatus) :
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def image_callback(self, image):
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        # self.boundingBoxes.image.data = sum(self.color_image, dim=0)
        #print(self.color_image)
        
        results = self.model(self.color_image)
        # xmin    ymin    xmax   ymax  confidence  class    name
        boxs = results.pandas().xyxy[0].values
        bev, lp, rp, lane_quality = calculate_curvature(self.color_image, boxs)
        # cv2.imshow('bev', bev)
        
        
        self.boundingBoxes.l_lane_curvation = np.round(np.float32(lp), 4)
        self.boundingBoxes.r_lane_curvation = np.round(np.float32(rp), 4)
        self.boundingBoxes.lane_quality = np.int32(lane_quality)

        global cnt
        cnt += 1
        cv2.imwrite('/root/catkin_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_ros/media/laneDect'+str(cnt)+'.png', bev)
        self.dectshow(self.color_image, boxs, image.height, image.width)

        cv2.waitKey(3)

    def classify_traffic_light(self, img_rgb, box):
        cropped_img = img_rgb[box.ymin:box.ymax, box.xmin:box.xmax]

        # 빨간색 범위
        lower_red = (0, 0, 155)
        upper_red = (160, 160, 255)

        # 노란색 범위
        lower_yellow = (0, 155, 155)
        upper_yellow = (180, 255, 255)

        # 초록색 범위
        lower_green = (0, 150, 0)
        upper_green = (180, 255, 180)

        # 색상 범위에 해당하는 마스크 생성
        mask_red = cv2.inRange(cropped_img, lower_red, upper_red)
        mask_yellow = cv2.inRange(cropped_img, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(cropped_img, lower_green, upper_green)

        # red_area = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_red)
        # yellow_area = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_yellow)
        # green_area = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_green)
        # cv2.imwrite('/root/catkin_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_ros/media/mask_green'+'.png', green_area)
        
        max_area = max(cv2.countNonZero(mask_red), cv2.countNonZero(mask_yellow), cv2.countNonZero(mask_green))
        if max_area == cv2.countNonZero(mask_red):
            box.Class = "red traffic light"
        elif max_area == cv2.countNonZero(mask_yellow):
            box.Class = "yellow traffic light"
        elif max_area == cv2.countNonZero(mask_green):
            box.Class = "green traffic light"
        
        return box.Class

    def dectshow(self, org_img, boxs, height, width):
        img = org_img.copy()
        
        cls_list = ['truck', 'car', 'traffic light', 'stop sign', 'bus', 'person']
        for box in boxs:
            boundingBox = BoundingBox()
            if box[-1] in cls_list:
                boundingBox.probability =np.float64(box[4])
                boundingBox.xmin = np.int64(box[0])
                boundingBox.ymin = np.int64(box[1])
                boundingBox.xmax = np.int64(box[2])
                boundingBox.ymax = np.int64(box[3])
                boundingBox.Class = box[-1]
                
                # classify traffic light 
                if boundingBox.Class == 'traffic light':
                    boundingBox.Class = self.classify_traffic_light(img, boundingBox)
                if box[-1] in self.classes_colors.keys():
                    color = self.classes_colors[box[-1]]
                else:
                    color = np.random.randint(0, 183, 3)
                    self.classes_colors[box[-1]] = color

                cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (int(color[0]),int(color[1]), int(color[2])), 2)

                if box[1] < 20:
                    text_pos_y = box[1] + 30
                else:
                    text_pos_y = box[1] - 10
                    
                cv2.putText(img, boundingBox.Class,
                            (int(box[0]), int(text_pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                self.boundingBoxes.bounding_boxes.append(boundingBox)
        
        self.position_pub.publish(self.boundingBoxes)

        self.publish_image(img, height, width)
        
        # cv2.imwrite('/root/catkin_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/laneDect'+str(cnt)+'.png', img)
        # cv2.imshow('YOLOv5_L', img)

    
   

    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        # header.frame_id = self.camera_frame
        header.frame_id = "left_camera"
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3
        self.l_image_pub.publish(image_temp)

def main():
    rospy.init_node('L_img_node', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":

    main()


# if __name__ == '__main__':

#     client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#     ip=socket.gethostbyname("203.246.114.249")
#     port=2210
#     address=(ip,port)
#     client.connect(address)
#     print("connection complete!!!!!!")
    
#     data = client.recv(64)
