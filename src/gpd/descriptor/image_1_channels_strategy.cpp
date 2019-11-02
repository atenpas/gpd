#include <gpd/descriptor/image_1_channels_strategy.h>

namespace gpd {
namespace descriptor {

std::vector<std::unique_ptr<cv::Mat>> Image1ChannelsStrategy::createImages(
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

void Image1ChannelsStrategy::createImage(const util::PointList &point_list,
                                         const candidate::Hand &hand,
                                         cv::Mat &image) const {
  // 1. Transform points and normals in neighborhood into the unit image.
  const Eigen::Matrix3Xd points = point_list.getPoints();
  const Eigen::Matrix3d rot = hand.getFrame().transpose();
  Eigen::Matrix3Xd points_frame =
      rot * (points - hand.getSample().replicate(1, points.cols()));
  std::vector<int> indices = findPointsInUnitImage(hand, points_frame);
  points_frame = transformPointsToUnitImage(hand, points_frame, indices);

  // 2. Calculate grasp image.
  Eigen::VectorXi cell_indices = findCellIndices(points_frame);
  image = createDepthImage(points_frame, cell_indices);

  if (is_plotting_) {
    std::string title = "Grasp Image (1 channel)";
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::Mat image_rgb;
    cvtColor(image, image_rgb, cv::COLOR_GRAY2RGB);
    cv::imshow(title, image_rgb);
    cv::waitKey(0);
  }
}

}  // namespace descriptor
}  // namespace gpd
