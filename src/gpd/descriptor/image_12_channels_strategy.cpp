#include <gpd/descriptor/image_12_channels_strategy.h>

namespace gpd {
namespace descriptor {

std::vector<std::unique_ptr<cv::Mat>> Image12ChannelsStrategy::createImages(
    const candidate::HandSet &hand_set,
    const util::PointList &nn_points) const {
  const std::vector<std::unique_ptr<candidate::Hand>> &hands =
      hand_set.getHands();
  std::vector<std::unique_ptr<cv::Mat>> images(hands.size());

  for (int i = 0; i < hands.size(); i++) {
    if (hand_set.getIsValid()(i)) {
      images[i] = std::make_unique<cv::Mat>(
          image_params_.size_, image_params_.size_,
          CV_8UC(image_params_.num_channels_), cv::Scalar(0.0));
      createImage(nn_points, *hands[i], *images[i]);
    }
  }

  return images;
}

void Image12ChannelsStrategy::createImage(const util::PointList &point_list,
                                          const candidate::Hand &hand,
                                          cv::Mat &image) const {
  // 1. Transform points and normals in neighborhood into the unit image.
  Matrix3XdPair points_normals = transformToUnitImage(point_list, hand);

  // 2. Create grasp image.
  image = calculateImage(points_normals.first, points_normals.second);
}

cv::Mat Image12ChannelsStrategy::calculateImage(
    const Eigen::Matrix3Xd &points, const Eigen::Matrix3Xd &normals) const {
  double t = omp_get_wtime();
  const int kNumProjections = 3;

  std::vector<cv::Mat> channels(image_params_.num_channels_);

  Eigen::Matrix3Xd points_proj = points;
  int swap_indices[3][2] = {{-1, -1}, {0, 2}, {1, 2}};
  // int swap_indices[3][2] = {{-1, -1}, {0, 2}, {0, 1}};

  for (size_t i = 0; i < kNumProjections; i++) {
    if (i > 0) {
      points_proj.row(swap_indices[i][0])
          .swap(points_proj.row(swap_indices[i][1]));
    }

    std::vector<cv::Mat> channels_i = calculateChannels(points_proj, normals);
    for (size_t j = 0; j < channels_i.size(); j++) {
      channels[i * 4 + j] = channels_i[j];
    }
  }

  cv::Mat image;
  cv::merge(channels, image);

  t = omp_get_wtime() - t;
  // printf("runtime: %3.10f\n", t);

  if (is_plotting_) {
    showImage(image);
  }

  return image;
}

std::vector<cv::Mat> Image12ChannelsStrategy::calculateChannels(
    const Eigen::Matrix3Xd &points, const Eigen::Matrix3Xd &normals) const {
  std::vector<cv::Mat> channels(4);

  Eigen::VectorXi cell_indices = findCellIndices(points);
  cv::Mat normals_image = createNormalsImage(normals, cell_indices);
  std::vector<cv::Mat> normals_image_channels;
  cv::split(normals_image, normals_image_channels);
  for (size_t i = 0; i < normals_image.channels(); i++) {
    channels[i] = normals_image_channels[i];
  }

  channels[3] = createDepthImage(points, cell_indices);

  return channels;
}

void Image12ChannelsStrategy::showImage(const cv::Mat &image) const {
  int border = 5;
  int n = 3;  // number of images in each row
  int m = 2;  // number of images in each column
  int image_size = image_params_.size_;
  int height = n * (image_size + border) + border;
  int width = m * (image_size + border) + border;
  cv::Mat image_out(height, width, CV_8UC3, cv::Scalar(0.5));
  std::vector<cv::Mat> channels;
  cv::split(image, channels);

  for (int i = 0; i < n; i++) {
    cv::Mat normals_rgb, depth_rgb;
    std::vector<cv::Mat> normals_channels(3);
    for (int j = 0; j < normals_channels.size(); j++) {
      normals_channels[j] = channels[i * 4 + j];
    }
    cv::merge(normals_channels, normals_rgb);
    // OpenCV requires images to be in BGR or grayscale to be displayed.
    cvtColor(normals_rgb, normals_rgb, cv::COLOR_RGB2BGR);
    cvtColor(channels[i * 4 + 3], depth_rgb, cv::COLOR_GRAY2RGB);
    normals_rgb.copyTo(image_out(cv::Rect(
        border, border + i * (border + image_size), image_size, image_size)));
    depth_rgb.copyTo(image_out(cv::Rect(2 * border + image_size,
                                        border + i * (border + image_size),
                                        image_size, image_size)));
  }

  cv::namedWindow("Grasp Image (12 channels)", cv::WINDOW_NORMAL);
  cv::imshow("Grasp Image (12 channels)", image_out);
  cv::waitKey(0);
}

}  // namespace descriptor
}  // namespace gpd
