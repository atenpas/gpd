# Grasp Pose Detection (GPD)

* [Author's website](http://www.ccs.neu.edu/home/atp/)
* [License](https://github.com/atenpas/gpd/blob/master/LICENSE.md)
* [ROS wrapper](https://github.com/atenpas/gpd_ros/)

Grasp Pose Detection (GPD) is a package to detect 6-DOF grasp poses (3-DOF
position and 3-DOF orientation) for a 2-finger robot hand (e.g., a parallel
jaw gripper) in 3D point clouds. GPD takes a point cloud as input and produces
pose estimates of viable grasps as output. The main strengths of GPD are:
- works for novel objects (no CAD models required for detection),
- works in dense clutter, and
- outputs 6-DOF grasp poses (enabling more than just top-down grasps).


<a href="http://www.youtube.com/watch?feature=player_embedded&v=kfe5bNt35ZI
" target="_blank"><img src="readme/ur5_video.jpg"
alt="UR5 demo" width="320" height="240" border="0" /></a>

GPD consists of two main steps: sampling a large number of grasp candidates, and classifying these candidates as viable grasps or not.

##### Example Input and Output
<img src="readme/clutter.png" height=170px/>

The reference for this package is:
[Grasp Pose Detection in Point Clouds](http://arxiv.org/abs/1706.09911).

## Table of Contents
1. [Requirements](#requirements)
1. [Installation](#install)
1. [Generate Grasps for a Point Cloud File](#pcd)
1. [Parameters](#parameters)
1. [Views](#views)
1. [Input Channels for Neural Network](#cnn_channels)
1. [CNN Frameworks](#cnn_frameworks)
1. [Network Training](#net_train)
1. [Grasp Image](#descriptor)
1. [References](#References)
1. [Troubleshooting](#troubleshooting)

<a name="requirements"></a>
## 1) Requirements

1. [PCL 1.9 or newer](http://pointclouds.org/)
2. [Eigen 3.0 or newer](https://eigen.tuxfamily.org)
3. [OpenCV 3.4 or newer](https://opencv.org)

<a name="install"></a>
## 2) Installation

The following instructions have been tested on **Ubuntu 16.04**. Similar
instructions should work for other Linux distributions.

1. Install [PCL](http://pointclouds.org/) and
[Eigen](https://eigen.tuxfamily.org). If you have ROS Indigo or Kinetic
installed, you should be good to go.

2. Install OpenCV 3.4 ([tutorial](https://www.python36.com/how-to-install-opencv340-on-ubuntu1604/)).

3. Clone the repository into some folder:

   ```
   git clone https://github.com/atenpas/gpd
   ```

4. Build the package:

   ```
   cd gpd
   mkdir build && cd build
   cmake ..
   make -j
   ```

You can optionally install GPD with `sudo make install` so that it can be used by other projects as a shared library.

If building the package does not work, try to modify the compiler flags, `CMAKE_CXX_FLAGS`, in the file CMakeLists.txt.

<a name="pcd"></a>
## 3) Generate Grasps for a Point Cloud File

Run GPD on an point cloud file (PCD or PLY):

   ```
   ./detect_grasps ../cfg/eigen_params.cfg ../tutorials/krylon.pcd
   ```

The output should look similar to the screenshot shown below. The window is the PCL viewer. You can press [q] to close the window and [h] to see a list of other commands.

<img src="readme/file.png" alt="" width="30%" border="0" />

Below is a visualization of the convention that GPD uses for the grasp pose (position and orientation) of a grasp. The grasp position is indicated by the orange cross and the orientation by the colored arrows.

<img src="readme/hand_frame.png" alt="" width="30%" border="0" />

<a name="parameters"></a>
## 4) Parameters

Brief explanations of parameters are given in [cfg/eigen_params.cfg](cfg/eigen_params.cfg).

The two parameters that you typically want to play with to **improve the
number of grasps found** are *workspace* and *num_samples*. The first defines the
volume of space in which to search for grasps as a cuboid of dimensions [minX,
maxX, minY, maxY, minZ, maxZ], centered at the origin of the point cloud frame.
The second is the number of samples that are drawn from the point cloud to
detect grasps. You should set the workspace as small as possible and the number
of samples as large as possible.

Most of the code is parallelized. To **improve runtime**, set *num_threads* to 
the number of (physical) CPU cores that your computer has available.

<a name="views"></a>
## 5) Views

![rviz screenshot](readme/views.png "Single View and Two Views")

You can use this package with a single or with two depth sensors. The package
comes with CAFFE model files for both. You can find these files in
*models/caffe/15channels*. For a single sensor, use
*single_view_15_channels.caffemodel* and for two depth sensors, use
*two_views_15_channels_[angle]*. The *[angle]* is the angle between the two
sensor views, as illustrated in the picture below. In the two-views setting, you
want to register the two point clouds together before sending them to GPD.

Providing the camera position to the configuration file (*.cfg) is important,
as it enables PCL to estimate the correct normals direction (which is to point
toward the camera). Alternatively, using the
[ROS wrapper](https://github.com/atenpas/gpd_ros/), multiple camera positions
can be provided.

![rviz screenshot](readme/view_angle.png "Angle Between Sensor Views")

To switch between one and two sensor views, change the parameter `weight_file`
in your config file.

<a name="cnn_channels"></a>
## 6) Input Channels for Neural Network

The package comes with weight files for two different input representations for
the neural network that is used to decide if a grasp is viable or not: 3 or 15
channels. The default is 15 channels. However, you can use the 3 channels to
achieve better runtime for a loss in grasp quality. For more details, please see
the references below.

<a name="cnn_frameworks"></a>
## 7) CNN Frameworks

GPD comes with a number of different classifier frameworks that
exploit different hardware and have different dependencies. Switching
between the frameworks requires to run CMake with additional arguments.
For example, to use the OpenVino framework:

   ```
   cmake .. -DUSE_OPENVINO=ON
   ```

You can use `ccmake` to check out all possible CMake options.

GPD supports the following three frameworks:

1. [OpenVino](https://software.intel.com/en-us/openvino-toolkit): [installation instructions](https://github.com/opencv/dldt/blob/2018/inference-engine/README.md) for open source version
(CPUs, GPUs, FPGAs from Intel)
1. [Caffe](https://caffe.berkeleyvision.org/) (GPUs from Nvidia or CPUs)
1. Custom LeNet implementation using the Eigen library (CPU)

Additional classifiers can be added by sub-classing the `classifier` interface.

##### OpenVINO

OpenVINO is **recommended for speed**. To use OpenVINO, you need to run the following command before compiling GPD.

   ```
   export InferenceEngine_DIR=/path/to/dldt/inference-engine/build/
   ```

<a name="net_train"></a>
## 8) Network Training

To create training data with the C++ code, you need to install [OpenCV 3.4 Contribs](https://www.python36.com/how-to-install-opencv340-on-ubuntu1604/).
Next, you need to compile GPD with the flag `DBUILD_DATA_GENERATION` like this:

    ```
    cd gpd
    mkdir build && cd build
    cmake .. -DBUILD_DATA_GENERATION=ON
    make -j
    ```

There are four steps to train a network to predict grasp poses. First, we need to create grasp images.

   ```
   ./generate_data ../cfg/generate_data.cfg
   ```

You should modify `generate_data.cfg` according to your needs.

Next, you need to resize the created databases to `train_offset` and `test_offset` (see the terminal output of `generate_data`). For example, to resize the training set, use the following commands with `size` set to the value of `train_offset`.
   ```
   cd pytorch
   python reshape_hdf5.py pathToTrainingSet.h5 out.h5 size
   ```

The third step is to train a neural network. The easiest way to training the network is with the existing code. This requires the **pytorch** framework. To train a network, use the following commands.

   ```
   cd pytorch
   python train_net3.py pathToTrainingSet.h5 pathToTestSet.h5 num_channels
   ```

The fourth step is to convert the model to the ONNX format.

   ```
   python torch_to_onxx.py pathToPytorchModel.pwf pathToONNXModel.onnx num_channels
   ```

The last step is to convert the ONNX file to an OpenVINO compatible format: [tutorial](https://software.intel.com/en-us/articles/OpenVINO-Using-ONNX#inpage-nav-4). This gives two files that can be loaded with GPD by modifying the `weight_file` and `model_file` parameters in a CFG file.

<a name="descriptor"></a>
## 9) Grasp Image/Descriptor
Generate some grasp poses and their corresponding images/descriptors:

   ```
   ./test_grasp_image ../tutorials/krylon.pcd 3456 1 ../models/lenet/15channels/params/
   ```

<img src="readme/image_15channels.png" alt="" width="30%" border="0" />

For details on how the grasp image is created, check out our [journal paper](http://arxiv.org/abs/1706.09911).

<a name="references"></a>
## 10) References

If you like this package and use it in your own work, please cite our journal
paper [1]. If you're interested in the (shorter) conference version, check out
[2].

[1] Andreas ten Pas, Marcus Gualtieri, Kate Saenko, and Robert Platt. [**Grasp
Pose Detection in Point Clouds**](http://arxiv.org/abs/1706.09911). The
International Journal of Robotics Research, Vol 36, Issue 13-14, pp. 1455-1473.
October 2017.

[2] Marcus Gualtieri, Andreas ten Pas, Kate Saenko, and Robert Platt. [**High
precision grasp pose detection in dense
clutter**](http://arxiv.org/abs/1603.01564). IROS 2016, pp. 598-605.

<a name="troubleshooting"></a>
## 11) Troubleshooting Tips

1. Remove the `cmake` cache: `rm CMakeCache.txt`
1. `make clean`
1. Remove the `build` folder and rebuild.
1. Update *gcc* and *g++* to a version > 5.
