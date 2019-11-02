#include <gpd/descriptor/image_strategy.h>

#include <gpd/descriptor/image_12_channels_strategy.h>
#include <gpd/descriptor/image_15_channels_strategy.h>
#include <gpd/descriptor/image_1_channels_strategy.h>
#include <gpd/descriptor/image_3_channels_strategy.h>

namespace gpd {
namespace descriptor {

std::unique_ptr<ImageStrategy> ImageStrategy::makeImageStrategy(
    const ImageGeometry &image_params, int num_threads, int num_orientations,
    bool is_plotting) {
  std::unique_ptr<ImageStrategy> strategy;
  if (image_params.num_channels_ == 1) {
    strategy = std::make_unique<Image1ChannelsStrategy>(
        image_params, num_threads, num_orientations, is_plotting);
  } else if (image_params.num_channels_ == 3) {
    strategy = std::make_unique<Image3ChannelsStrategy>(
        image_params, num_threads, num_orientations, is_plotting);
  } else if (image_params.num_channels_ == 12) {
    strategy = std::make_unique<Image12ChannelsStrategy>(
        image_params, num_threads, num_orientations, is_plotting);
  } else if (image_params.num_channels_ == 15) {
    strategy = std::make_unique<Image15ChannelsStrategy>(
        image_params, num_threads, num_orientations, is_plotting);
  }

  return strategy;
}

Matrix3XdPair ImageStrategy::transformToUnitImage(
    const util::PointList &point_list, const candidate::Hand &hand) const {
  // 1. Transform points and normals in neighborhood into the hand frame.
  const Eigen::Matrix3Xd rotation = hand.getFrame().transpose();
  const Eigen::Vector3d &sample = hand.getSample();
  Matrix3XdPair points_normals(
      rotation *
          (point_list.getPoints() - sample.replicate(1, point_list.size())),
      rotation * point_list.getNormals());

  // 2. Find points in unit image.
  const std::vector<int> indices =
      findPointsInUnitImage(hand, points_normals.first);
  points_normals.first =
      transformPointsToUnitImage(hand, points_normals.first, indices);
  points_normals.second =
      util::EigenUtils::sliceMatrix(points_normals.second, indices);

  return points_normals;
}

std::vector<int> ImageStrategy::findPointsInUnitImage(
    const candidate::Hand &hand, const Eigen::Matrix3Xd &points) const {
  std::vector<int> indices;
  const double half_outer_diameter = image_params_.outer_diameter_ / 2.0;

  for (int i = 0; i < points.cols(); i++) {
    if ((points(0, i) > hand.getBottom()) &&
        (points(0, i) < hand.getBottom() + image_params_.depth_) &&
        (points(1, i) > hand.getCenter() - half_outer_diameter) &&
        (points(1, i) < hand.getCenter() + half_outer_diameter) &&
        (points(2, i) > -1.0 * image_params_.height_) &&
        (points(2, i) < image_params_.height_)) {
      indices.push_back(i);
    }
  }

  return indices;
}

Eigen::Matrix3Xd ImageStrategy::transformPointsToUnitImage(
    const candidate::Hand &hand, const Eigen::Matrix3Xd &points,
    const std::vector<int> &indices) const {
  Eigen::Matrix3Xd points_out(3, indices.size());
  const double half_outer_diameter = image_params_.outer_diameter_ / 2.0;
  const double double_height = 2.0 * image_params_.height_;

  for (int i = 0; i < indices.size(); i++) {
    points_out(0, i) =
        (points(0, indices[i]) - hand.getBottom()) / image_params_.depth_;
    points_out(1, i) =
        (points(1, indices[i]) - (hand.getCenter() - half_outer_diameter)) /
        image_params_.outer_diameter_;
    points_out(2, i) =
        (points(2, indices[i]) + image_params_.height_) / double_height;
  }

  return points_out;
}

Eigen::VectorXi ImageStrategy::findCellIndices(
    const Eigen::Matrix3Xd &points) const {
  double cellsize = 1.0 / (double)image_params_.size_;
  const Eigen::VectorXi vertical_cells =
      (floorVector(points.row(0) / cellsize)).cwiseMin(image_params_.size_ - 1);
  const Eigen::VectorXi horizontal_cells =
      (floorVector(points.row(1) / cellsize)).cwiseMin(image_params_.size_ - 1);
  Eigen::VectorXi cell_indices =
      horizontal_cells + vertical_cells * image_params_.size_;
  return cell_indices;
}

cv::Mat ImageStrategy::createBinaryImage(
    const Eigen::VectorXi &cell_indices) const {
  cv::Mat image(image_params_.size_, image_params_.size_, CV_8UC1,
                cv::Scalar(0));

  for (int i = 0; i < cell_indices.rows(); i++) {
    const int &idx = cell_indices[i];
    int row = image.rows - 1 - idx / image.cols;
    int col = idx % image.cols;
    image.at<uchar>(row, col) = 255;
  }

  // Dilate the image to fill in holes.
  cv::Mat dilation_element =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  dilate(image, image, dilation_element);

  return image;
}

cv::Mat ImageStrategy::createNormalsImage(
    const Eigen::Matrix3Xd &normals,
    const Eigen::VectorXi &cell_indices) const {
  cv::Mat image(image_params_.size_, image_params_.size_, CV_32FC3,
                cv::Scalar(0.0));

  for (int i = 0; i < cell_indices.rows(); i++) {
    const int &idx = cell_indices[i];
    int row = image_params_.size_ - 1 - idx / image_params_.size_;
    int col = idx % image_params_.size_;
    cv::Vec3f &v = image.at<cv::Vec3f>(row, col);
    const Eigen::Vector3d &n = normals.col(i);
    if (v(0) == 0 && v(1) == 0 && v(2) == 0) {
      v = cv::Vec3f(fabs(n(0)), fabs(n(1)), fabs(n(2)));
    } else {
      v += (cv::Vec3f(fabs(n(0)), fabs(n(1)), fabs(n(2))) - v) *
           (1.0 / sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)));
    }
  }

  // Dilate the image to fill in holes.
  cv::Mat dilation_element =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  dilate(image, image, dilation_element);

  // Normalize the image to the range [0,1].
  cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC3);

  // Convert float image to uchar image, required by Caffe.
  image.convertTo(image, CV_8U, 255.0);

  return image;
}

cv::Mat ImageStrategy::createDepthImage(
    const Eigen::Matrix3Xd &points, const Eigen::VectorXi &cell_indices) const {
  cv::Mat image(image_params_.size_, image_params_.size_, CV_32FC1,
                cv::Scalar(0.0));
  float avgs[image_params_.size_ * image_params_.size_] = {0};
  float counts[image_params_.size_ * image_params_.size_] = {0};

  //  double t0 = omp_get_wtime();
  for (int i = 0; i < cell_indices.rows(); i++) {
    const int &idx = cell_indices[i];
    int row = image.rows - 1 - idx / image.cols;
    int col = idx % image.cols;
    counts[idx] += 1.0;
    avgs[idx] += (points(2, i) - avgs[idx]) * (1.0 / counts[idx]);
    float &v = image.at<float>(row, col);
    v = 1.0 - avgs[idx];
  }
  //  printf("average calc runtime: %3.8f\n", omp_get_wtime() - t0);
  //  t0 = omp_get_wtime();

  // Dilate the image to fill in holes.
  cv::Mat dilation_element =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  dilate(image, image, dilation_element);

  // Normalize the image to the range [0,1].
  cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);

  // Convert float image to uchar image, required by Caffe.
  image.convertTo(image, CV_8U, 255.0);
  //  printf("image ops runtime: %3.8f\n", omp_get_wtime() - t0);

  return image;
}

cv::Mat ImageStrategy::createShadowImage(
    const Eigen::Matrix3Xd &points, const Eigen::VectorXi &cell_indices) const {
  // Calculate average depth image.
  cv::Mat image(image_params_.size_, image_params_.size_, CV_32FC1,
                cv::Scalar(0.0));
  cv::Mat nonzero(image_params_.size_, image_params_.size_, CV_8UC1,
                  cv::Scalar(0));
  float counts[image_params_.size_ * image_params_.size_] = {0};

  for (int i = 0; i < cell_indices.rows(); i++) {
    const int &idx = cell_indices[i];
    int row = image.rows - 1 - idx / image.cols;
    int col = idx % image.cols;
    counts[idx] += 1.0;
    image.at<float>(row, col) +=
        (points(2, i) - image.at<float>(row, col)) * (1.0 / counts[idx]);
    nonzero.at<uchar>(row, col) = 1;
  }

  // Reverse depth so that closest points have largest value.
  double min, max;
  cv::Point min_loc, max_loc;
  cv::minMaxLoc(image, &min, &max, &min_loc, &max_loc, nonzero);
  cv::Mat max_img(image_params_.size_, image_params_.size_, CV_32FC1,
                  cv::Scalar(0.0));
  max_img.setTo(max, nonzero);
  image = max_img - image;

  // Dilate the image to fill in holes.
  cv::Mat dilation_element =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  dilate(image, image, dilation_element);

  // Normalize the image to the range [0,1].
  cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);

  // Convert float image to uchar image, required by Caffe.
  image.convertTo(image, CV_8U, 255.0);

  return image;
}

Eigen::VectorXi ImageStrategy::floorVector(const Eigen::VectorXd &a) const {
  Eigen::VectorXi b(a.size());

  for (int i = 0; i < b.size(); i++) {
    b(i) = floor((double)a(i));
  }

  return b;
}

}  // namespace descriptor
}  // namespace gpd
