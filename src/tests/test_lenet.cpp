#include <Eigen/Dense>

#include <gpd/net/lenet.h>

namespace gpd {
namespace test {
namespace {

int DoMain(int argc, char *argv[]) {
  // Test with input from file.
  net::Lenet network(
      1, "/home/atenpas/catkin_ws/src/gpd_no_ros/lenet/3channels/params/", 60,
      15);
  std::vector<float> x = network.readFileLineByLineIntoVector(
      "src/gpd/caffe/15channels/txt/rand_input.txt");
  network.forward(x);

  return 0;
}

}  // namespace
}  // namespace test
}  // namespace gpd

int main(int argc, char *argv[]) { return gpd::test::DoMain(argc, argv); }
