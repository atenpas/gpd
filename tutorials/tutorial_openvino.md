# Tutorial Enable OpenVINO

In this tutorial, we introduce how to enable OpenVINO option for grasp detection.

## 1. Install OpenVINO toolkit
OpenVINO supports multiple host OS. We verified with Linux.
[Installing the OpenVINO Toolkit for Linux](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux)

## 2. Verify OpenVINO installation
Try to run the demo scripts.
[Verify the Demo Scripts](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux#inpage-nav-3-5)
Then try to run OpenVINO inference example applications.
[Build the Samples](https://software.intel.com/en-us/articles/OpenVINO-InferEngine#inpage-nav-6),
[Running the Samples](https://software.intel.com/en-us/articles/OpenVINO-InferEngine#inpage-nav-7)

## 3. Build GPD with OpenVINO
Setup OpenVINO environment variables, replacing <OPENVINO_INSTALL_DIR> with the specific location.
```
source <OPENVINO_INSTALL_DIR>/bin/setupvars.sh
```
Once OpenVINO installed, build GPD with option "USE_OPENVINO" ([OFF]|ON)
```
catkin_make -DCMAKE_BUILD_TYPE=Release -DUSE_OPENVINO=ON --pkg gpd
```

## 4. Launch GPD with OpenVINO
The launch process is similar to [Detect Grasps With an RGBD camera](tutorials/tutorial_1_grasps_camera.md),
just with an additional param "device" ([0:CPU]|1:GPU|2:VPU|3:FPGA) to specify the target device to execute the
grasp detection.
```
# launch the openni camera for pointcloud2
roslaunch openni2_launch openni2.launch
# start rviz
rosrun rviz rviz
# setup OpenVINO environment variables, replacing <OPENVINO_INSTALL_DIR> with the specific location
source <OPENVINO_INSTALL_DIR>/bin/setupvars.sh
# launch the grasp detection
roslaunch gpd tutorial1.launch device:=0
```
