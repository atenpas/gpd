#include <gpd/descriptor/image_15_channels_strategy.h>

namespace gpd {
namespace descriptor {

std::vector<std::unique_ptr<cv::Mat>> Image15ChannelsStrategy::createImages(
    const candidate::HandSet &hand_set,
    const util::PointList &nn_points) const {
  const std::vector<std::unique_ptr<candidate::Hand>> &hands =
      hand_set.getHands();
  std::vector<std::unique_ptr<cv::Mat>> images(hands.size());

  Eigen::Matrix3Xd shadow = hand_set.calculateShadow(nn_points, shadow_length_);

  for (int i = 0; i < hands.size(); i++) {
    if (hand_set.getIsValid()(i)) {
      images[i] = std::make_unique<cv::Mat>(
          image_params_.size_, image_params_.size_,
          CV_8UC(image_params_.num_channels_), cv::Scalar(0.0));
      createImage(nn_points, *hands[i], shadow, *images[i]);
    }
  }

  return images;
}

void Image15ChannelsStrategy::createImage(const util::PointList &point_list,
                                          const candidate::Hand &hand,
                                          const Eigen::Matrix3Xd &shadow,
                                          cv::Mat &image) const {
  // 1. Transform points and normals in neighborhood into the unit image.
  Matrix3XdPair points_normals = transformToUnitImage(point_list, hand);

  // 2. Transform occluded points into hand frame.
  Eigen::Matrix3Xd shadow_frame =
      shadow - hand.getSample().replicate(1, shadow.cols());
  shadow_frame = hand.getFrame().transpose() * shadow_frame;
  std::vector<int> indices = findPointsInUnitImage(hand, shadow_frame);
  Eigen::Matrix3Xd cropped_shadow_points =
      transformPointsToUnitImage(hand, shadow_frame, indices);

  // 3. Create grasp image.
  image = calculateImage(points_normals.first, points_normals.second,
                         cropped_shadow_points);
}

cv::Mat Image15ChannelsStrategy::calculateImage(
    const Eigen::Matrix3Xd &points, const Eigen::Matrix3Xd &normals,
    const Eigen::Matrix3Xd &shadow) const {
  double t = omp_get_wtime();
  const int kNumProjections = 3;

  std::vector<cv::Mat> channels(image_params_.num_channels_);

  Eigen::Matrix3Xd points_proj = points;
  Eigen::Matrix3Xd shadow_proj = shadow;
  int swap_indices[3][2] = {{-1, -1}, {0, 2}, {1, 2}};

  for (size_t i = 0; i < kNumProjections; i++) {
    if (i > 0) {
      points_proj.row(swap_indices[i][0])
          .swap(points_proj.row(swap_indices[i][1]));
      shadow_proj.row(swap_indices[i][0])
          .swap(shadow_proj.row(swap_indices[i][1]));
    }
    std::vector<cv::Mat> channels_i =
        calculateChannels(points_proj, normals, shadow_proj);
    for (size_t j = 0; j < channels_i.size(); j++) {
      channels[i * 5 + j] = channels_i[j];
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

std::vector<cv::Mat> Image15ChannelsStrategy::calculateChannels(
    const Eigen::Matrix3Xd &points, const Eigen::Matrix3Xd &normals,
    const Eigen::Matrix3Xd &shadow) const {
  std::vector<cv::Mat> channels(5);

  Eigen::VectorXi cell_indices = findCellIndices(points);
  cv::Mat normals_image = createNormalsImage(normals, cell_indices);
  std::vector<cv::Mat> normals_image_channels;
  cv::split(normals_image, normals_image_channels);
  for (size_t i = 0; i < normals_image.channels(); i++) {
    channels[i] = normals_image_channels[i];
  }

  channels[3] = createDepthImage(points, cell_indices);

  cell_indices = findCellIndices(shadow);
  channels[4] = createShadowImage(shadow, cell_indices);

  return channels;
}

void Image15ChannelsStrategy::showImage(const cv::Mat &image) const {
  int border = 5;
  int n = 3;
  int image_size = image_params_.size_;
  int total_size = n * (image_size + border) + border;
  cv::Mat image_out(total_size, total_size, CV_8UC3, cv::Scalar(0.5));
  std::vector<cv::Mat> channels;
  cv::split(image, channels);

  for (int i = 0; i < n; i++) {
    // OpenCV requires images to be in BGR or grayscale to be displayed.
    cv::Mat normals_rgb, depth_rgb, shadow_rgb;
    std::vector<cv::Mat> normals_channels(3);
    for (int j = 0; j < normals_channels.size(); j++) {
      normals_channels[j] = channels[i * 5 + j];
    }
    cv::merge(normals_channels, normals_rgb);
    // OpenCV requires images to be in BGR or grayscale to be displayed.
    cvtColor(normals_rgb, normals_rgb, cv::COLOR_RGB2BGR);
    cvtColor(channels[i * 5 + 3], depth_rgb, cv::COLOR_GRAY2RGB);
    cvtColor(channels[i * 5 + 4], shadow_rgb, cv::COLOR_GRAY2RGB);
    normals_rgb.copyTo(image_out(cv::Rect(
        border, border + i * (border + image_size), image_size, image_size)));
    depth_rgb.copyTo(image_out(cv::Rect(2 * border + image_size,
                                        border + i * (border + image_size),
                                        image_size, image_size)));
    shadow_rgb.copyTo(image_out(cv::Rect(3 * border + 2 * image_size,
                                         border + i * (border + image_size),
                                         image_size, image_size)));
  }

  cv::namedWindow("Grasp Image (15 channels)", cv::WINDOW_NORMAL);
  cv::imshow("Grasp Image (15 channels)", image_out);
  cv::waitKey(0);
}

}  // namespace descriptor
}  // namespace gpd
