#include <Eigen/Dense>

#include <gpd/net/conv_layer.h>

namespace gpd {
namespace test {
namespace {

int DoMain(int argc, char *argv[]) {
  // Create example input, weights, bias.
  Eigen::MatrixXf X(5, 5);
  Eigen::MatrixXf W(3, 3);
  Eigen::VectorXf b = Eigen::VectorXf::Zero(1);
  X << 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
      0;
  W << 1, 0, 1, 0, 1, 0, 1, 0, 1;

  std::vector<float> w_vec;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      W_rowmajor(W);
  Eigen::Map<Eigen::VectorXf> w(W_rowmajor.data(), W_rowmajor.size());
  w_vec.assign(w.data(), w.data() + w.size());

  std::vector<float> b_vec;
  b_vec.assign(b.data(), b.data() + b.size());

  // Create a convolutional layer and execute a forward pass.
  net::ConvLayer conv1(5, 5, 1, 1, 3, 1, 0);
  conv1.setWeightsAndBiases(w_vec, b_vec);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      X_row_major(X);
  Eigen::Map<Eigen::VectorXf> v1(X_row_major.data(), X_row_major.size());
  std::vector<float> vec1;
  vec1.assign(v1.data(), v1.data() + v1.size());
  Eigen::MatrixXf Y = conv1.forward(vec1);

  std::cout << "Y: " << Y.rows() << " x " << Y.cols() << std::endl;
  std::cout << Y << std::endl;
  std::cout << std::endl;
}

}  // namespace
}  // namespace test
}  // namespace gpd

int main(int argc, char *argv[]) { return gpd::test::DoMain(argc, argv); }
