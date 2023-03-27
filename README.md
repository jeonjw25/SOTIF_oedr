# SOTIF OEDR Model

OEDR model for camera SOTIF process

<br>

# Quick Start
## 1. Set up execution environment
- Prerequestions
  - ubuntu 18.04 ROS melodic
  - cuda = 11.0
  - RTX 30xx Ti
  
  <br>

- Install yolov5 dependencies
  - move to `Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_ros/yolov5`
  - Install dependencies
  ```
  $ cd ./Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_ros/yolov5
  $ pip3 install -r requirements.txt
  ```
  - If there are additional dependencies, Install them

<br>

- Matching torch version dependency with cuda
```
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

<br>

- Install bagfiles from [here](https://drive.google.com/file/d/1wmsllgCpF-djAhiN5Hgz4Bs-bqeZQPYe/view?usp=sharing)
  - Put a bagfile to  `Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_ros/scripts`


<br>

- build & execute launch file
```
$ cm
$ roslaunch yolov5_ros oedr.launch
```

<br>

- Output topic received in `output_recv.py`

<br>

# Docker 

- pull Image and make container 
```
$ docker run -it --name your container name \
-v your workspace path://root/catkin_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_ros \
-v /tmp/.X11-unix:/tmp/.X11-unix \
--net host \
-e DISPLAY=$DISPLAY \
--gpus all jeonjw25/sotif_oedr:v1
```
- Execute launch file
```
$ roslaunch yolov5_ros oedr.launch
```
<br>

# System archtecture

![image](https://user-images.githubusercontent.com/54730375/227845567-554e5925-fc7f-42bd-b102-1ae30feaaf24.png)

<br>

# Execution nodes in launchfile

![image](https://user-images.githubusercontent.com/54730375/227845806-fe86010d-80f5-4983-a41e-fb8c5b3bb080.png)

<br>

# rqt graph

![image](https://user-images.githubusercontent.com/54730375/227845865-f706eba5-38db-4d5b-85b5-e0269e05ee61.png)