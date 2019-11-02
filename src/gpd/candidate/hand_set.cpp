#include <gpd/candidate/hand_set.h>

#include <random>

namespace gpd {
namespace candidate {

const Eigen::Vector3d HandSet::AXES[3] = {Eigen::Vector3d::UnitX(),
                                          Eigen::Vector3d::UnitY(),
                                          Eigen::Vector3d::UnitZ()};

const bool HandSet::MEASURE_TIME = false;

int HandSet::seed_ = 0;

HandSet::HandSet(const HandGeometry &hand_geometry,
                 const Eigen::VectorXd &angles,
                 const std::vector<int> &hand_axes, int num_finger_placements,
                 bool deepen_hand, Antipodal &antipodal)
    : hand_geometry_(hand_geometry),
      angles_(angles),
      hand_axes_(hand_axes),
      num_finger_placements_(num_finger_placements),
      deepen_hand_(deepen_hand),
      antipodal_(antipodal) {
  sample_.setZero();
  hands_.resize(0);
  is_valid_.resize(0);
}

void HandSet::evalHandSet(const util::PointList &point_list,
                          const LocalFrame &local_frame) {
  hands_.resize(hand_axes_.size() * angles_.size());
  is_valid_ = Eigen::Array<bool, 1, Eigen::Dynamic>::Constant(
      1, angles_.size() * hand_axes_.size(), false);

  // Local reference frame
  sample_ = local_frame.getSample();
  frame_ << local_frame.getNormal(), local_frame.getBinormal(),
      local_frame.getCurvatureAxis();

  // Iterate over rotation axes.
  for (int i = 0; i < hand_axes_.size(); i++) {
    int start = i * angles_.size();
    evalHands(point_list, local_frame, hand_axes_[i], start);
  }
}

void HandSet::evalHands(const util::PointList &point_list,
                        const LocalFrame &local_frame, int axis, int start) {
  // Rotate about binormal by 180 degrees to reverses direction of normal.
  const Eigen::Matrix3d ROT_BINORMAL =
      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()).toRotationMatrix();

  // This object is used to evaluate the finger placement.
  FingerHand finger_hand(hand_geometry_.finger_width_,
                         hand_geometry_.outer_diameter_, hand_geometry_.depth_,
                         num_finger_placements_);

  // Set the forward and lateral axis of the robot hand frame (closing direction
  // and grasp approach direction).
  finger_hand.setForwardAxis(0);
  finger_hand.setLateralAxis(1);

  // Evaluate grasp at each hand orientation.
  for (int i = 0; i < angles_.rows(); i++) {
    // Rotation about <axis> by <angles_(i)> radians.
    Eigen::Matrix3d rot =
        Eigen::AngleAxisd(angles_(i), AXES[axis]).toRotationMatrix();

    // Rotate points into this hand orientation.
    Eigen::Matrix3d frame_rot;
    frame_rot.noalias() = frame_ * ROT_BINORMAL * rot;
    util::PointList point_list_frame = point_list.transformToHandFrame(
        local_frame.getSample(), frame_rot.transpose());

    // Crop points on hand height.
    util::PointList point_list_cropped =
        point_list_frame.cropByHandHeight(hand_geometry_.height_);

    // Evaluate finger placements for this orientation.
    finger_hand.evaluateFingers(point_list_cropped.getPoints(),
                                hand_geometry_.init_bite_);

    // Check that there is at least one feasible 2-finger placement.
    finger_hand.evaluateHand();

    // Create the grasp candidate.
    hands_[start + i] = std::make_unique<Hand>(local_frame.getSample(),
                                               frame_rot, finger_hand, 0.0);

    // Check that there is at least one feasible 2-finger placement.
    if (finger_hand.getHand().any()) {
      int finger_idx;
      if (deepen_hand_) {
        // Try to move the hand as deep as possible onto the object.
        finger_idx = finger_hand.deepenHand(point_list_cropped.getPoints(),
                                            hand_geometry_.init_bite_,
                                            hand_geometry_.depth_);
      } else {
        finger_idx = finger_hand.chooseMiddleHand();
      }
      // Calculate points in the closing region of the hand.
      std::vector<int> indices_closing =
          finger_hand.computePointsInClosingRegion(
              point_list_cropped.getPoints(), finger_idx);
      if (indices_closing.size() == 0) {
        continue;
      }

      is_valid_[start + i] = true;
      modifyCandidate(*hands_[start + i], point_list_cropped, indices_closing,
                      finger_hand);
    }
  }
}

Eigen::Matrix3Xd HandSet::calculateShadow(const util::PointList &point_list,
                                          double shadow_length) const {
  // Set voxel size for points that fill occluded region.
  const double voxel_grid_size = 0.003;
  // Calculate number of points along each shadow vector.
  double num_shadow_points = floor(shadow_length / voxel_grid_size);

  const int num_cams = point_list.getCamSource().rows();

  Eigen::Matrix3Xd shadow;

  // Calculate the set of cameras which see the points.
  Eigen::VectorXi camera_set = point_list.getCamSource().rowwise().sum();

  // Calculate the center point of the point neighborhood.
  Eigen::Vector3d center = point_list.getPoints().rowwise().sum();
  center /= (double)point_list.size();

  // Stores the list of all bins of the voxelized, occluded points.
  std::vector<Vector3iSet> shadows;
  shadows.resize(num_cams, Vector3iSet(num_shadow_points * 10000));

  for (int i = 0; i < num_cams; i++) {
    if (camera_set(i) >= 1) {
      double t0_if = omp_get_wtime();

      // Calculate the unit vector that points from the camera position to the
      // center of the point neighborhood.
      Eigen::Vector3d shadow_vec = center - point_list.getViewPoints().col(i);

      // Scale that vector by the shadow length.
      shadow_vec = shadow_length * shadow_vec / shadow_vec.norm();

      // Calculate occluded points for this camera.
      calculateShadowForCamera(point_list.getPoints(), shadow_vec,
                               num_shadow_points, voxel_grid_size, shadows[i]);
    }
  }

  // Only one camera view point.
  if (num_cams == 1) {
    // Convert voxels back to points.
    shadow = shadowVoxelsToPoints(
        std::vector<Eigen::Vector3i>(shadows[0].begin(), shadows[0].end()),
        voxel_grid_size);
    return shadow;
  }

  // Multiple camera view points: find the intersection of all sets of occluded
  // points.
  double t0_intersection = omp_get_wtime();
  Vector3iSet bins_all = shadows[0];

  for (int i = 1; i < num_cams; i++) {
    // Check that there are points seen by this camera.
    if (camera_set(i) >= 1) {
      bins_all = intersection(bins_all, shadows[i]);
    }
  }
  if (MEASURE_TIME) {
    printf("intersection runtime: %.3fs\n", omp_get_wtime() - t0_intersection);
  }

  // Convert voxels back to points.
  std::vector<Eigen::Vector3i> voxels(bins_all.begin(), bins_all.end());
  shadow = shadowVoxelsToPoints(voxels, voxel_grid_size);
  return shadow;
}

Eigen::Matrix3Xd HandSet::shadowVoxelsToPoints(
    const std::vector<Eigen::Vector3i> &voxels, double voxel_grid_size) const {
  // Convert voxels back to points.
  double t0_voxels = omp_get_wtime();
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> distr{0.0, 1.0};
  Eigen::Matrix3Xd shadow(3, voxels.size());

  for (int i = 0; i < voxels.size(); i++) {
    shadow.col(i) =
        voxels[i].cast<double>() * voxel_grid_size +
        Eigen::Vector3d::Ones() * distr(gen) * voxel_grid_size * 0.3;
  }
  if (MEASURE_TIME) {
    printf("voxels-to-points runtime: %.3fs\n", omp_get_wtime() - t0_voxels);
  }

  return shadow;
}

void HandSet::calculateShadowForCamera(const Eigen::Matrix3Xd &points,
                                       const Eigen::Vector3d &shadow_vec,
                                       int num_shadow_points,
                                       double voxel_grid_size,
                                       Vector3iSet &shadow_set) const {
  double t0_set = omp_get_wtime();
  const int n = points.cols() * num_shadow_points;
  const double voxel_grid_size_mult = 1.0 / voxel_grid_size;
  const double max = 1.0 / 32767.0;

  for (int i = 0; i < n; i++) {
    const int pt_idx = i / num_shadow_points;
    shadow_set.insert(
        ((points.col(pt_idx) + ((double)fastrand() * max) * shadow_vec) *
         voxel_grid_size_mult)
            .cast<int>());
  }

  if (MEASURE_TIME) {
    printf(
        "Shadow (1 camera) calculation. Runtime: %.3f, #points: %d, "
        "num_shadow_points: %d, #shadow: %d, max #shadow: %d\n",
        omp_get_wtime() - t0_set, (int)points.cols(), num_shadow_points,
        (int)shadow_set.size(), n);
  }
}

void HandSet::modifyCandidate(Hand &hand, const util::PointList &point_list,
                              const std::vector<int> &indices,
                              const FingerHand &finger_hand) const {
  // Modify the grasp.
  hand.construct(finger_hand);

  // Extract points in hand closing region.
  util::PointList point_list_closing = point_list.slice(indices);

  // Calculate grasp width (hand opening width).
  double width = point_list_closing.getPoints().row(1).maxCoeff() -
                 point_list_closing.getPoints().row(1).minCoeff();
  hand.setGraspWidth(width);

  // Evaluate if the grasp is antipodal.
  labelHypothesis(point_list_closing, finger_hand, hand);
}

void HandSet::labelHypothesis(const util::PointList &point_list,
                              const FingerHand &finger_hand, Hand &hand) const {
  int label =
      antipodal_.evaluateGrasp(point_list, 0.003, finger_hand.getLateralAxis(),
                               finger_hand.getForwardAxis(), 2);
  hand.setHalfAntipodal(label == Antipodal::HALF_GRASP ||
                        label == Antipodal::FULL_GRASP);
  hand.setFullAntipodal(label == Antipodal::FULL_GRASP);
}

inline int HandSet::fastrand() const {
  seed_ = (214013 * seed_ + 2531011);
  return (seed_ >> 16) & 0x7FFF;
}

Vector3iSet HandSet::intersection(const Vector3iSet &set1,
                                  const Vector3iSet &set2) const {
  if (set2.size() < set1.size()) {
    return intersection(set2, set1);
  }

  Vector3iSet set_out(set1.size());

  for (Vector3iSet::const_iterator it = set1.begin(); it != set1.end(); it++) {
    if (set2.find(*it) != set2.end()) {
      set_out.insert(*it);
    }
  }

  return set_out;
}

}  // namespace candidate
}  // namespace gpd
