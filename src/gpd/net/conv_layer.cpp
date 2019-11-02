#include <gpd/net/conv_layer.h>

namespace gpd {
namespace net {

ConvLayer::ConvLayer(int width, int height, int depth, int num_filters,
                     int spatial_extent, int stride, int padding)
    : w1(width),
      h1(height),
      d1(depth),
      d2(num_filters),
      f(spatial_extent),
      s(stride) {
  w2 = (w1 - spatial_extent + 2 * padding) / stride + 1;
  h2 = (h1 - spatial_extent + 2 * padding) / stride + 1;

  // matrix X_col has size: spatial_extent*spatial_extent*d1 x w2*h2
  X_col_r = spatial_extent * spatial_extent * d1;
  X_col_c = w2 * h2;

  // matrix W_row has size: num_filters x patial_extent*spatial_extent*d1
  W_row_r = num_filters;
  W_row_c = spatial_extent * spatial_extent * d1;
}

Eigen::MatrixXf ConvLayer::forward(const std::vector<float> &x) const {
  // Convert input image to matrix where each column is an image patch.
  std::vector<float> x_col_vec;
  x_col_vec.resize(X_col_r * X_col_c);
  imageToColumns(&x[0], d1, h1, w1, d2, f, f, s, s, &x_col_vec[0]);
  Eigen::MatrixXf X_col =
      ConvLayer::RowMajorMatrixMap(x_col_vec.data(), X_col_r, X_col_c);

  // Convert weights vector to matrix where each row is a kernel.
  Eigen::MatrixXf W_row =
      ConvLayer::RowMajorMatrixMap(weights_.data(), W_row_r, W_row_c);

  // Convert biases array to Eigen vector.
  Eigen::VectorXf b =
      Eigen::Map<const Eigen::VectorXf>(biases_.data(), biases_.size());

  // Calculate the convolution by calculating the dot product of W_row and
  // X_col.
  Eigen::MatrixXf H = forward(W_row, b, X_col);
  return H;
}

Eigen::MatrixXf ConvLayer::forward(const Eigen::MatrixXf &W,
                                   const Eigen::VectorXf &b,
                                   const Eigen::MatrixXf &X) const {
  Eigen::MatrixXf B = b.replicate(1, X.cols());

  // Calculate the forward pass.
  Eigen::MatrixXf H = W * X + B;  // np.dot(W_row, X_col)
  return H;
}

bool ConvLayer::is_a_ge_zero_and_a_lt_b(int a, int b) const {
  return a >= 0 && a < b;
}

void ConvLayer::imageToColumns(const float *data_im, const int channels,
                               const int height, const int width,
                               const int num_kernels, const int kernel_h,
                               const int kernel_w, const int stride_h,
                               const int stride_w, float *data_col) const {
  const int output_h = (height - kernel_h) / stride_h + 1;
  const int output_w = (width - kernel_w) / stride_w + 1;
  const int channel_size = height * width;

  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = kernel_row;

        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = kernel_col;

            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

}  // namespace net
}  // namespace gpd
