import cv2
import numpy as np

def draw_lines(img, lines):
    #print("draw_lines funcion")
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], 2)

def warp(img, src, dst):
    img_size = (img.shape[1], img.shape[0]) # w h
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (1000,800), flags=cv2.INTER_LINEAR)
    
    return warped

def color_filter(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])

    yellow_lower = np.uint8([10, 0, 100])
    yellow_upper = np.uint8([40, 255, 255])

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, lower, upper)

    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(img, img, mask=mask)

    return mask
    

def roi(img):
    x = int(img.shape[1])
    y = int(img.shape[0])
    _shape = np.array([[0, y-1], [0, 10], [int(0.3*x), 10], [int(0.3*x), int(0.3*y)], [int(0.7*x), int(0.3*y)], [int(0.7*x), 10], [x, 10], [x, y-1]]) 

    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_cnt = img.shape[2]
        ignore_mask_color = (255,) * channel_cnt
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img

def plothistogram(img):
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint

    return leftbase, rightbase

def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    nwindows = 15
    window_height = np.int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    margin = 80
    minpix = 50

    left_lane = []
    right_lane = []

    color = [0, 255, 0]
    thickness = 2

    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w+1) * window_height # top of window
        win_y_high = binary_warped.shape[0] - w * window_height # bottom of window
        if left_current - margin > 0:
            win_xleft_low = left_current - margin # left top of left window
        else:
            win_xleft_low = 0
        win_xleft_high = left_current + margin # right bottom of left window
        win_xright_low = right_current - margin # left top of right window
        if right_current + margin <= binary_warped.shape[1]:
            win_xright_high = right_current + margin # right bottom of right window
        else:
            win_xright_high = binary_warped.shape[1]

        # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        good_left = ((nonzero_y >=win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >=win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        
        
        left_lane.append(good_left)
        right_lane.append(good_right)
        
        # cv2.imshow("oo" ,out_img)

        if len(good_left) > minpix:
            left_current = np.int(1.02 * np.mean(nonzero_x[good_left]))
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        if len(good_right) > minpix:
            right_current = np.int(1.02 * np.mean(nonzero_x[good_right]))
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)

    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]
    m_leftx, m_lefty, m_rightx, m_righty = convert_pixel2meter(binary_warped, leftx, lefty, rightx, righty)

    if len(m_leftx) == 0 and len(m_rightx) > 0: # turning left
        
        right_fit = np.polyfit(m_righty, m_rightx, 3)
        left_fit = right_fit
        print("left turning!")
    elif len(m_leftx) > 0 and len(m_rightx) == 0: # turning right
        left_fit = np.polyfit(m_lefty, m_leftx, 3)
        right_fit = left_fit
        print("right turning!")
    else:
        right_fit = np.polyfit(m_righty, m_rightx, 3)
        left_fit = np.polyfit(m_lefty, m_leftx, 3)
    return out_img, left_fit, right_fit

def convert_pixel2meter(img, leftx, lefty, rightx, righty):
    ym_per_pix = 12.3 / img.shape[0]
    xm_per_pix = 4 / img.shape[1]
    left_x_mid = np.full((len(leftx)), int(img.shape[1] / 2))
    right_x_mid = int(img.shape[1] / 2)
    m_leftx = (leftx - left_x_mid) * xm_per_pix
    m_rightx = (rightx - right_x_mid) * xm_per_pix
    m_lefty = lefty * ym_per_pix
    m_righty = righty * ym_per_pix


    return m_leftx, m_lefty, m_rightx, m_righty

    
def calculate_curvature(image, boxes):

    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 이미지 크기 가져오기
    height, width = gray_img.shape[:2]

    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    bgr_blur_img = cv2.GaussianBlur(bgr_img, (3, 3), 0)

    # canny_img = cv2.Canny(blur_img, 70, 210)
    
    # 관심영역(ROI) 생성
    region_of_interest_vertices = np.array([[(0, height), (0, height-100), (width / 4, height / 2), (3*width / 4, height / 2), (width, height-100), (width, height)]], dtype='uint32')
    # src = np.float32([[530, height], [1030, 620], [1150, 620], [1650, height]])
    # dst = np.float32([[80, 800], [260, 0], [650, 0], [750, 800]])
    src = np.float32([[0, height], [750, 700], [1200, 700], [width, height]])
    dst = np.float32([[250, 750], [320, 50], [850, 50], [850, 750]])

    warped = warp(bgr_blur_img, src, dst) # birdeye view transform
    w_f_img = color_filter(warped) # filter yellow & white
    masked_w_f_img = roi(w_f_img) # filter middle area of lane
    canny_img = cv2.Canny(warped, 70, 210)

    leftbase, rightbase = plothistogram(masked_w_f_img)
    out_img, left_fit, right_fit = slide_window_search(masked_w_f_img, leftbase, rightbase)

    lp = np.poly1d(left_fit)
    rp = np.poly1d(right_fit)

    # print("left_curvation: {}" .format(left_fit))
    # print("right_curvation: {}" .format(right_fit))

    return out_img, lp, rp

    # 관심영역 외부의 부분을 모두 검은색으로 만들기
    # mask = np.zeros_like(gray_img)
    # cv2.fillPoly(mask, np.array([region_of_interest_vertices], np.int32), 255)
    # masked_image = cv2.bitwise_and(canny_img, mask)

    # crop vehicle area
    # crop_img = crop(masked_image, boxes)

    # line detection
    # cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # lines = cv2.HoughLinesP(canny_img, 1, np.pi/180, 30, np.array([]), minLineLength=5, maxLineGap=25)

    # slope filter
    # line_arr = np.squeeze(lines)
    # slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi
    # line_arr = line_arr[np.abs(slope_degree)<170]
    # slope_degree = slope_degree[np.abs(slope_degree)<170]
    # line_arr = line_arr[np.abs(slope_degree)>95]
    # slope_degree = slope_degree[np.abs(slope_degree)>95]
    # L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
    # L_lines, R_lines = L_lines[:,None], R_lines[:,None]
    
    # left_x = []
    # left_y = []
    # right_x = []
    # right_y = []
    # for i in np.squeeze(L_lines):
    #     # print("L_points: {}" .format(i))
    #     left_x.append(i[0])
    #     left_y.append(i[1])

    # for j in np.squeeze(R_lines):
    #     # print("L_points: {}" .format(i))
    #     right_x.append(j[0])
    #     right_y.append(j[1])
    
    # left_fit = np.polyfit(left_y, left_x, 3)
    # right_fit = np.polyfit(right_y, right_x, 3)
    

    # draw_lines(warped, R_lines)
    # draw_lines(warped, L_lines)
    
    # # 차량과 차선의 거리 계산
    # lane_width = right_point - left_point
    # car_position = mid_point
    # lane_center_position = (left_point + right_point) / 2
    # distance_from_center = (car_position - lane_center_position) * 3.7 / lane_width
    
    # # 곡률 계산
    # y_eval = height
    # left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
    # right_curvature = ((1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])

    

def crop(org_img, boxes):
    img = org_img.copy()
    # print(img.shape)
    for box in boxes:
        xmin = np.int64(box[0])
        ymin = np.int64(box[1])
        xmax = np.int64(box[2])
        ymax = np.int64(box[3])
        img[ymin:ymax, xmin:xmax] = 0
        

    return img

    
