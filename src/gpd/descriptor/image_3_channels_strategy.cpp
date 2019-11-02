#include <gpd/descriptor/image_3_channels_strategy.h>

namespace gpd {
namespace descriptor {

std::vector<std::unique_ptr<cv::Mat>> Image3ChannelsStrategy::createImages(
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

void Image3ChannelsStrategy::createImage(const util::PointList &point_list,
                                         const candidate::Hand &hand,
                                         cv::Mat &image) const {
  // 1. Transform points and normals in neighborhood into the unit image.
  Matrix3XdPair points_normals = transformToUnitImage(point_list, hand);

  // 2. Calculate grasp image.
  Eigen::VectorXi cell_indices = findCellIndices(points_normals.first);
  image = createNormalsImage(points_normals.second, cell_indices);

  if (is_plotting_) {
    std::string title = "Grasp Image (3 channels)";
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, image);
    cv::waitKey(0);
  }
}

}  // namespace descriptor
}  // namespace gpd
