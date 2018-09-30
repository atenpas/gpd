# Tutorial Enable OpenVINO™ (Open Visual Inference & Neural Network Optimization)

In this tutorial, we introduce how to enable [OpenVINO™](https://software.intel.com/en-us/openvino-toolkit)
option for grasp detection. Based on convolutional neural networks (CNN),
the toolkit extends workloads across Intel® hardware and maximizes performance.

## 1. Install OpenVINO™ toolkit
OpenVINO supports multiple host OS. We verified with 64-bit Ubuntu 16.04.

For gpd execution at Intel CPU, please follow the
[Installing the OpenVINO™ Toolkit for Linux](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux).

For gpd execution at Intel GPU, please follow additional steps to install OpenCL NEO driver
[Additional Installation Steps for Processor Graphics (GPU)](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux#inpage-nav-4-1).

For gpd execution at Intel Movidius NCS, please follow additional steps to configure UDEV rules.
[Additional Installation Steps for the Intel® Movidius™ Neural Compute Stick](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux#inpage-nav-4-2).

## 2. Verify OpenVINO™ installation
Try to run the demo scripts: [Verify the Demo Scripts](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux#inpage-nav-3-5)

Then try to run OpenVINO inference example applications:
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
