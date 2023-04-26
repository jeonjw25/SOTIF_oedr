# SOTIF OEDR Model

OEDR model for camera SOTIF process

<br>

# Quick Start(Local)

- Prerequestions
  - ubuntu 18.04 ROS melodic
  - cuda = 11.0
  - RTX 30xx Ti
  
  <br>
- Git clone this repo to ROS melodic catkin_ws/src/

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

- Install bagfiles from [here](https://drive.google.com/file/d/1wmsllgCpF-djAhiN5Hgz4Bs-bqeZQPYe/view?usp=sharing) and [here](https://drive.google.com/file/d/1FzOU2kbddMhqZ1drE-abUewXbCTNIq7-/view?usp=sharing)
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

- Link docker to local xhost
```
xhost +local:docker
```

<br>

- Install cuda container toolkit and add user to the docker group.
```sh
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2

# Add your user to the docker group.
$ sudo usermod -aG docker $USER
$ sudo systemctl restart docker
```
- Restart or relogin

<br>

- Pull Image and make container 
```sh
$ docker run -it --name your_container_name \
-v your_workspace_path:/root/catkin_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_ros \
-v /tmp/.X11-unix:/tmp/.X11-unix \
--net host \
-e DISPLAY=$DISPLAY \
--gpus all jeonjw25/cuda11.2_cudnn8_ubuntu18.04_ros:v1
```

<br>

## in docker CLI

- Appy package setup 
```sh
$ source ~/catkin_ws/devel/setup.bash
```
<br>

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