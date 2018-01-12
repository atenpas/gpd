#include <Eigen/Dense>

#include <gpd/lenet.h>


int main(int argc, char* argv[])
{
  // Test with input from file.
  Lenet net(1, "/home/andreas/projects/grasp_pose_detection/catkin_ws/src/gpd/caffe/15channels/bin/");
  std::vector<float> x = net.readFileLineByLineIntoVector("src/gpd/caffe/15channels/txt/rand_input.txt");
  net.forward(x);

  return 0;
}
