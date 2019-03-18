# Tutorial Enable Intel® OpenVINO™

**Amazing, same software execution on various vision devices: CPU, GPU, VPU, FPGA, ...**

In this tutorial, we introduce how to enable OpenVINO™ (Open Visual Inference & Neural Network Optimization) option for grasp detection.

Intel® Distribution of OpenVINO™ (Open Visual Inference & Neural Network Optimization) toolkit, based on convolutional neural networks (CNN), extends workloads across Intel® hardware (including accelerators) and maximizes performance. The toolkit enables CNN-based deep learning inference at the edge computation, and supports heterogeneous execution across various compution vision devices -- CPU, GPU, Intel® Movidius™ NCS, and FPGA -- using a **common** API.

The toolkit is available from open source project [Intel OpenVINO Toolkit](https://github.com/opencv/dldt). Refer to the online installation guides to build and install Inference Engine for [Linux](https://github.com/opencv/dldt/blob/2018/inference-engine/README.md#build-on-linux-systems), or follow the below detailed steps we have validated successfully.

## Install OpenVINO toolkit
* Build and install Inference Engine
   ```bash
   git clone https://github.com/opencv/dldt.git
   cd dldt/inference-engine
   git submodule init
   git submodule update --recursive
   # install common dependencies
   source ./install_dependencies.sh
   # install mkl for cpu acceleration
   wget https://github.com/intel/mkl-dnn/releases/download/v0.17/mklml_lnx_2019.0.1.20180928.tgz
   tar -zxvf mklml_lnx_2019.0.1.20180928.tgz
   sudo ln -s `pwd`/mklml_lnx_2019.0.1.20180928 /usr/local/lib/mklml
   # install opencl for gpu acceleration
   wget https://github.com/intel/compute-runtime/releases/download/18.28.11080/intel-opencl_18.28.11080_amd64.deb
   sudo dpkg -i intel-opencl_18.28.11080_amd64.deb
   # build
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DGEMM=MKL -DMKLROOT=/usr/local/lib/mklml -DENABLE_MKL_DNN=ON -DENABLE_CLDNN=ON ..
   make -j8
   sudo make install
   ```
* Share the CMake configures for Inference Engine to be found by other packages
   ```bash
   sudo mkdir /usr/share/InferenceEngine
   sudo cp InferenceEngineConfig*.cmake /usr/share/InferenceEngine
   sudo cp targets.cmake /usr/share/InferenceEngine
   ```
   Then Inference Engine will be found when adding "find_package(InferenceEngine)" into the CMakeLists.txt
* Configure library path for dynamic loading
   ```bash
   echo `pwd`/../bin/intel64/Release/lib | sudo tee -a /etc/ld.so.conf.d/openvino.conf
   sudo ldconfig
   ```
* Optionally install plug-ins for InferenceEngine deployment on heterogeneous devices

  Install [plug-in](https://software.intel.com/en-us/neural-compute-stick/get-started) for deployment on Intel Movidius Neural Computation Sticks Myriad X.

## Build GPD with OpenVINO
Once OpenVINO installed, build GPD with option "USE_OPENVINO".
```
cd ~/catkin_ws/src
git clone https://github.com/atenpas/gpd.git
cd ~/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release -DUSE_OPENVINO=ON --pkg gpd
```
### Build Options
* USE_OPENVINO (**OFF**|ON) Switch on/off the usage of OpenVINO

## Launch GPD with OpenVINO
The launch process is similar to [Detect Grasps With an RGBD camera](tutorials/tutorial_1_grasps_camera.md),
just with an additional launch option "device".
```
# launch the openni camera for pointcloud2
roslaunch openni2_launch openni2.launch
# start rviz
rosrun rviz rviz -d src/gpd/tutorials/openni2.rviz
# launch the grasp detection. "device:=0" for CPU, "device:=1" for GPU, "device:=2" for VPU
roslaunch gpd tutorial1.launch device:=0
```
### Launch Options
* device (**0:CPU**|1:GPU|2:VPU|3:FPGA) Specify the target device to execute the grasp detection inference
