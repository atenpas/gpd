#include <gpd/clustering.h>

namespace gpd {

std::vector<std::unique_ptr<candidate::Hand>> Clustering::findClusters(
    const std::vector<std::unique_ptr<candidate::Hand>> &hand_list,
    bool remove_inliers) {
  // const double AXIS_ALIGN_ANGLE_THRESH = 15.0 * M_PI/180.0;
  const double AXIS_ALIGN_ANGLE_THRESH = 12.0 * M_PI / 180.0;
  const double AXIS_ALIGN_DIST_THRESH = 0.005;
  // const double MAX_DIST_THRESH = 0.07;
  const double MAX_DIST_THRESH = 0.05;
  //  const int max_inliers = 50;

  std::vector<std::unique_ptr<candidate::Hand>> hands_out;
  std::vector<bool> has_used;
  if (remove_inliers) {
    has_used.resize(hand_list.size());
    for (int i = 0; i < hand_list.size(); i++) {
      has_used[i] = false;
    }
  }

  std::vector<int> inliers;

  for (int i = 0; i < hand_list.size(); i++) {
    int num_inliers = 0;
    Eigen::Vector3d position_delta = Eigen::Vector3d::Zero();
    Eigen::Matrix3d axis_outer_prod =
        hand_list[i]->getAxis() * hand_list[i]->getAxis().transpose();
    inliers.resize(0);
    double mean = 0.0;
    double standard_deviation = 0.0;

    for (int j = 0; j < hand_list.size(); j++) {
      if (i == j || (remove_inliers && has_used[j])) continue;

      // Which hands have an axis within <AXIS_ALIGN_ANGLE_THRESH> of this one?
      double axis_aligned =
          hand_list[i]->getAxis().transpose() * hand_list[j]->getAxis();
      bool axis_aligned_binary =
          fabs(axis_aligned) > cos(AXIS_ALIGN_ANGLE_THRESH);

      // Which hands are within <MAX_DIST_THRESH> of this one?
      Eigen::Vector3d delta_pos =
          hand_list[i]->getPosition() - hand_list[j]->getPosition();
      double delta_pos_mag = delta_pos.norm();
      bool delta_pos_mag_binary = delta_pos_mag <= MAX_DIST_THRESH;

      // Which hands are within <AXIS_ALIGN_DIST_THRESH> of this one when
      // projected onto the plane orthognal to this
      // one's axis?
      Eigen::Matrix3d axis_orth_proj =
          Eigen::Matrix3d::Identity() - axis_outer_prod;
      Eigen::Vector3d delta_pos_proj = axis_orth_proj * delta_pos;
      double delta_pos_proj_mag = delta_pos_proj.norm();
      bool delta_pos_proj_mag_binary =
          delta_pos_proj_mag <= AXIS_ALIGN_DIST_THRESH;

      bool inlier_binary = axis_aligned_binary && delta_pos_mag_binary &&
                           delta_pos_proj_mag_binary;
      if (inlier_binary) {
        inliers.push_back(i);
        num_inliers++;
        position_delta += hand_list[j]->getPosition();
        double old_mean = mean;
        mean += (hand_list[j]->getScore() - mean) /
                static_cast<double>(num_inliers);
        standard_deviation += (hand_list[j]->getScore() - mean) *
                              (hand_list[j]->getScore() - old_mean);
        // mean += hand_list[j]->getScore();
        // standard_deviation +=
        //     hand_list[j]->getScore() * hand_list[j]->getScore();
        if (remove_inliers) {
          has_used[j] = true;
        }
      }
    }

    if (num_inliers >= min_inliers_) {
      double dNumInliers = static_cast<double>(num_inliers);
      position_delta =
          position_delta / dNumInliers - hand_list[i]->getPosition();
      standard_deviation /= dNumInliers;
      if (standard_deviation != 0) {
        standard_deviation = sqrt(standard_deviation);
      }
      double sqrt_num_inliers = sqrt((double)num_inliers);
      double conf_lb = mean - 2.576 * standard_deviation / sqrt_num_inliers;
      double conf_ub = mean + 2.576 * standard_deviation / sqrt_num_inliers;
      printf("grasp %d, inliers: %d, ||position_delta||: %3.4f, ", i,
             num_inliers, position_delta.norm());
      printf("mean: %3.4f, STD: %3.4f, conf_int: (%3.4f, %3.4f)\n", mean,
             standard_deviation, conf_lb, conf_ub);
      std::unique_ptr<candidate::Hand> hand =
          std::make_unique<candidate::Hand>(*hand_list[i]);
      hand->setPosition(hand->getPosition() + position_delta);
      hand->setScore(conf_lb);
      hand->setFullAntipodal(hand_list[i]->isFullAntipodal());
      hands_out.push_back(std::move(hand));
    }
  }

  return hands_out;
}

}  // namespace gpd
