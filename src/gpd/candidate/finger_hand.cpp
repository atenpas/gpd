#include <gpd/candidate/finger_hand.h>

namespace gpd {
namespace candidate {

FingerHand::FingerHand(double finger_width, double hand_outer_diameter,
                       double hand_depth, int num_placements)
    : finger_width_(finger_width),
      hand_depth_(hand_depth),
      lateral_axis_(-1),
      forward_axis_(-1) {
  // Calculate the finger spacing.
  Eigen::VectorXd fs_half;
  fs_half.setLinSpaced(num_placements, 0.0, hand_outer_diameter - finger_width);
  finger_spacing_.resize(2 * num_placements);
  finger_spacing_
      << (fs_half.array() - hand_outer_diameter + finger_width_).matrix(),
      fs_half;

  fingers_ = Eigen::Array<bool, 1, Eigen::Dynamic>::Constant(
      1, 2 * num_placements, false);
  hand_ =
      Eigen::Array<bool, 1, Eigen::Dynamic>::Constant(1, num_placements, false);
}

void FingerHand::evaluateFingers(const Eigen::Matrix3Xd &points, double bite,
                                 int idx) {
  // Calculate top and bottom of the hand (top = fingertip, bottom = base).
  top_ = bite;
  bottom_ = bite - hand_depth_;

  center_ = 0.0;

  fingers_.setConstant(false);

  // Crop points at bite.
  std::vector<int> cropped_indices;
  for (int i = 0; i < points.cols(); i++) {
    if (points(forward_axis_, i) < bite) {
      // Check that the hand would be able to extend by <bite> onto the object
      // without causing the back of the hand to
      // collide with <points>.
      if (points(forward_axis_, i) < bottom_) {
        return;
      }

      cropped_indices.push_back(i);
    }
  }

  // Check that there is at least one point in between the fingers.
  if (cropped_indices.size() == 0) {
    return;
  }

  // Identify free gaps (finger placements that do not collide with the point
  // cloud).
  if (idx == -1) {
    for (int i = 0; i < fingers_.size(); i++) {
      if (isGapFree(points, cropped_indices, i)) {
        fingers_(i) = true;
      }
    }
  } else {
    if (isGapFree(points, cropped_indices, idx)) {
      fingers_(idx) = true;
    }

    if (isGapFree(points, cropped_indices, fingers_.size() / 2 + idx)) {
      fingers_(fingers_.size() / 2 + idx) = true;
    }
  }
}

void FingerHand::evaluateHand() {
  const int n = fingers_.size() / 2;

  for (int i = 0; i < n; i++) {
    hand_(i) = (fingers_(i) == true && fingers_(n + i) == true);
  }
}

void FingerHand::evaluateHand(int idx) {
  const int n = fingers_.size() / 2;
  hand_.setConstant(false);
  hand_(idx) = (fingers_(idx) == true && fingers_(n + idx) == true);
}

int FingerHand::chooseMiddleHand() {
  std::vector<int> hand_idx;

  for (int i = 0; i < hand_.cols(); i++) {
    if (hand_(i) == true) {
      hand_idx.push_back(i);
    }
  }

  if (hand_idx.size() == 0) {
    return -1;
  }

  int idx = hand_idx[ceil(hand_idx.size() / 2.0) - 1];

  return idx;
}

int FingerHand::deepenHand(const Eigen::Matrix3Xd &points, double min_depth,
                           double max_depth) {
  // Choose middle hand.
  int hand_eroded_idx = chooseMiddleHand();  // middle index
  int opposite_idx =
      fingers_.size() / 2 + hand_eroded_idx;  // opposite finger index

  // Attempt to deepen hand (move as far onto the object as possible without
  // collision).
  const double DEEPEN_STEP_SIZE = 0.005;
  FingerHand new_hand = *this;
  FingerHand last_new_hand = new_hand;

  for (double depth = min_depth + DEEPEN_STEP_SIZE; depth <= max_depth;
       depth += DEEPEN_STEP_SIZE) {
    // Check if the new hand placement is feasible
    new_hand.evaluateFingers(points, depth, hand_eroded_idx);
    if (!new_hand.fingers_(hand_eroded_idx) ||
        !new_hand.fingers_(opposite_idx)) {
      break;
    }

    hand_(hand_eroded_idx) = true;
    last_new_hand = new_hand;
  }

  // Recover the deepest hand.
  *this = last_new_hand;
  hand_.setConstant(false);
  hand_(hand_eroded_idx) = true;

  return hand_eroded_idx;
}

std::vector<int> FingerHand::computePointsInClosingRegion(
    const Eigen::Matrix3Xd &points, int idx) {
  // Find feasible finger placement.
  if (idx == -1) {
    for (int i = 0; i < hand_.cols(); i++) {
      if (hand_(i) == true) {
        idx = i;
        break;
      }
    }
  }

  // Calculate the lateral parameters of the hand closing region for this finger
  // placement.
  left_ = finger_spacing_(idx) + finger_width_;
  right_ = finger_spacing_(hand_.cols() + idx);
  center_ = 0.5 * (left_ + right_);
  surface_ = points.row(lateral_axis_).minCoeff();

  // Find points inside the hand closing region defined by <bottom_>, <top_>,
  // <left_> and <right_>.
  std::vector<int> indices;
  for (int i = 0; i < points.cols(); i++) {
    if (points(forward_axis_, i) > bottom_ && points(forward_axis_, i) < top_ &&
        points(lateral_axis_, i) > left_ && points(lateral_axis_, i) < right_) {
      indices.push_back(i);
    }
  }

  return indices;
}

bool FingerHand::isGapFree(const Eigen::Matrix3Xd &points,
                           const std::vector<int> &indices, int idx) {
  for (int i = 0; i < indices.size(); i++) {
    const double &x = points(lateral_axis_, indices[i]);

    if (x > finger_spacing_(idx) && x < finger_spacing_(idx) + finger_width_) {
      return false;
    }
  }

  return true;
}

}  // namespace candidate
}  // namespace gpd
