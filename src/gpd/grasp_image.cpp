#include <gpd/grasp_image.h>


int GraspImage::image_size_;


GraspImage::GraspImage(int image_size, int num_channels, bool is_plotting)
  : num_channels_(num_channels), is_plotting_(is_plotting)
{
  GraspImage::image_size_ = image_size;
}


Eigen::VectorXi GraspImage::findCellIndices(const Eigen::Matrix3Xd& points)
{
  double cellsize = 1.0 / (double) image_size_;
  Eigen::VectorXi vertical_cells = (floorVector(points.row(0) / cellsize)).cwiseMin(image_size_ - 1);
  Eigen::VectorXi horizontal_cells = (floorVector(points.row(1) / cellsize)).cwiseMin(image_size_ - 1);
  Eigen::VectorXi cell_indices = horizontal_cells + vertical_cells * image_size_;
  return cell_indices;
}


cv::Mat GraspImage::createBinaryImage(const Eigen::VectorXi& cell_indices)
{
  cv::Mat image(image_size_, image_size_, CV_8UC1, cv::Scalar(0));

  // Calculate average depth image.
  for (int i = 0; i < cell_indices.rows(); i++)
  {
    const int& idx = cell_indices[i];
    int row = image.rows - 1 - idx / image.cols;
    int col = idx % image.cols;
    image.at<uchar>(row, col) = 255;
  }

  // Dilate the image to fill in holes.
  cv::Mat dilation_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
  dilate(image, image, dilation_element);

  return image;
}


cv::Mat GraspImage::createNormalsImage(const Eigen::Matrix3Xd& normals, const Eigen::VectorXi& cell_indices)
{
  // For each cell, calculate the average surface normal of the points that fall into that cell.
  cv::Mat image(image_size_, image_size_, CV_32FC3, cv::Scalar(0.0));

  for (int i = 0; i < cell_indices.rows(); i++)
  {
    const int& idx = cell_indices[i];
    int row = image_size_ - 1 - idx / image_size_;
    int col = idx % image_size_;
    cv::Vec3f& v = image.at<cv::Vec3f>(row, col);
    const Eigen::Vector3d& n = normals.col(i);
    if (v(0) == 0 && v(1) == 0 && v(2) == 0)
    {
      v = cv::Vec3f(fabs(n(0)), fabs(n(1)), fabs(n(2)));
    }
    else
    {
      v += (cv::Vec3f(fabs(n(0)), fabs(n(1)), fabs(n(2))) - v) * (1.0/sqrt(v(0)*v(0) + v(1)*v(1) + v(2)*v(2)));
    }
  }

  // Dilate the image to fill in holes.
  cv::Mat dilation_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
  dilate(image, image, dilation_element);

  // Normalize the image to the range [0,1].
  cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC3);

  // Convert float image to uchar image, required by Caffe.
  image.convertTo(image, CV_8U, 255.0);

  return image;
}


cv::Mat GraspImage::createDepthImage(const Eigen::Matrix3Xd& points, const Eigen::VectorXi& cell_indices)
{
  cv::Mat image(image_size_, image_size_, CV_32FC1, cv::Scalar(0.0));
  float avgs[image_size_ * image_size_]; // average of cell
  float counts[image_size_ * image_size_]; // count of cell

  for (int i = 0; i < image_size_ * image_size_; i++)
  {
    avgs[i] = 0.0;
    counts[i] = 0;
  }

  // Calculate average depth image.
  for (int i = 0; i < cell_indices.rows(); i++)
  {
    const int& idx = cell_indices[i];
    int row = image.rows - 1 - idx / image.cols;
    int col = idx % image.cols;
    counts[idx] += 1.0;
    avgs[idx] += (points(2,i) - avgs[idx]) * (1.0/counts[idx]);
    float& v = image.at<float>(row, col);
    v = 1.0 - avgs[idx];
  }

  // Dilate the image to fill in holes.
  cv::Mat dilation_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
  dilate(image, image, dilation_element);

  // Normalize the image to the range [0,1].
  cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);

  // Convert float image to uchar image, required by Caffe.
  image.convertTo(image, CV_8U, 255.0);

  return image;
}


cv::Mat GraspImage::createShadowImage(const Eigen::Matrix3Xd& points, const Eigen::VectorXi& cell_indices)
{
  // Calculate average depth image.
  cv::Mat image(image_size_, image_size_, CV_32FC1, cv::Scalar(0.0));
  cv::Mat nonzero(image_size_, image_size_, CV_8UC1, cv::Scalar(0));
  float counts[image_size_ * image_size_];

  for (int i = 0; i < image_size_ * image_size_; i++)
  {
    counts[i] = 0;
  }

  for (int i = 0; i < cell_indices.rows(); i++)
  {
    const int& idx = cell_indices[i];
    int row = image.rows - 1 - idx / image.cols;
    int col = idx % image.cols;
    counts[idx] += 1.0;
    image.at<float>(row, col) += (points(2,i) - image.at<float>(row, col)) * (1.0/counts[idx]);
    nonzero.at<uchar>(row, col) = 1;
  }

  // Reverse depth so that closest points have largest value.
  double min, max;
  cv::Point min_loc, max_loc;
  cv::minMaxLoc(image, &min, &max, &min_loc, &max_loc, nonzero);
  cv::Mat max_img(image_size_, image_size_, CV_32FC1, cv::Scalar(0.0));
  max_img.setTo(max, nonzero);
  image = max_img - image;

  // Dilate the image to fill in holes.
  cv::Mat dilation_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
  dilate(image, image, dilation_element);

  // Normalize the image to the range [0,1].
  cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);

  // Convert float image to uchar image, required by Caffe.
  image.convertTo(image, CV_8U, 255.0);

  return image;
}


Eigen::VectorXi GraspImage::floorVector(const Eigen::VectorXd& a)
{
  Eigen::VectorXi b(a.size());

  for (int i = 0; i < b.size(); i++)
  {
    b(i) = floor((double) a(i));
  }

  return b;
}
