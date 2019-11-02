#include <gpd/net/dense_layer.h>

namespace gpd {
namespace net {

Eigen::MatrixXf DenseLayer::forward(const std::vector<float> &x) const {
  Eigen::Map<const Eigen::MatrixXf> W(weights_.data(), num_units_, x.size());
  Eigen::Map<const Eigen::VectorXf> b(biases_.data(), biases_.size());
  Eigen::Map<const Eigen::VectorXf> X(x.data(), x.size());

  // Calculate the forward pass.
  Eigen::MatrixXf H = W * X + b;  // np.dot(W_row, X_col)

  return H;
}

}  // namespace net
}  // namespace gpd
