import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

class calRelativeVal:
    img_left1 = [376, 739]
    img_left2 = [563, 674]
    img_left3 = [658, 642]

    img_mid1 = [1261, 735]
    img_mid2 = [1159, 676]
    img_mid3 = [1110, 641]

    img_right1 = [1664, 739]
    img_right2 = [1438, 674]
    img_right3 = [1320, 641]

    obj_left1 = [12.6, 3.5, 0]
    obj_left2 = [18.6, 3.5, 0]
    obj_left3 = [24.6, 3.5, 0]

    obj_mid1 = [12.6, -1.75, 0]
    obj_mid2 = [18.6, -1.75, 0]
    obj_mid3 = [24.6, -1.75, 0]

    obj_right1 = [12.6, -4.25, 0]
    obj_right2 = [18.6, -4.25, 0]
    obj_right3 = [24.6, -4.25, 0]

    img_points = np.array([img_left1, img_left2, img_left3, img_mid1, img_mid2, img_mid3, img_right1, img_right2, img_right3], dtype=np.float32)
    obj_points = np.array([obj_left1, obj_left2, obj_left3, obj_mid1, obj_mid2, obj_mid3, obj_right1, obj_right2, obj_right3], dtype=np.float32)

    
    H, _ = cv2.findHomography(img_points, obj_points)
    appned_image_points = np.append(img_points.reshape(9, 2), np.ones([1, 9]).T, axis=1)
    print(appned_image_points)
    for image_point in appned_image_points:
    # estimation point(object_point) -> homography * src(image_point)
        estimation_distance = np.dot(H, image_point)
        x, y, z = estimation_distance
        print("x: {}, y: {}".format(round(x/z, 2), round(y/z, 2)))
    # img = cv2.imread("homography.png")
    # plt.imshow(img)
    # plt.show()
