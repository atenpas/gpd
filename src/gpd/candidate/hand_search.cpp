#include <gpd/candidate/hand_search.h>

namespace gpd {
namespace candidate {

const int HandSearch::ROTATION_AXIS_NORMAL = 0;
const int HandSearch::ROTATION_AXIS_BINORMAL = 1;
const int HandSearch::ROTATION_AXIS_CURVATURE_AXIS = 2;

HandSearch::HandSearch(Parameters params)
    : params_(params), plots_local_axes_(false) {
  // Calculate radius for nearest neighbor search.
  const HandGeometry &hand_geom = params_.hand_geometry_;
  Eigen::Vector3d hand_dims;
  hand_dims << hand_geom.outer_diameter_ - hand_geom.finger_width_,
      hand_geom.depth_, hand_geom.height_ / 2.0;
  nn_radius_ = hand_dims.maxCoeff();
  antipodal_ =
      std::make_unique<Antipodal>(params.friction_coeff_, params.min_viable_);
  plot_ = std::make_unique<util::Plot>(params_.hand_axes_.size(),
                                       params_.num_orientations_);
}

std::vector<std::unique_ptr<HandSet>> HandSearch::searchHands(
    const util::Cloud &cloud_cam) const {
  double t0_total = omp_get_wtime();

  // Create KdTree for neighborhood search.
  const PointCloudRGB::Ptr &cloud = cloud_cam.getCloudProcessed();
  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud);

  // 1. Estimate local reference frames.
  std::cout << "Estimating local reference frames ...\n";
  std::vector<LocalFrame> frames;
  FrameEstimator frame_estimator(params_.num_threads_);
  if (cloud_cam.getSamples().cols() > 0) {  // use samples
    frames = frame_estimator.calculateLocalFrames(
        cloud_cam, cloud_cam.getSamples(), params_.nn_radius_frames_, kdtree);
  } else if (cloud_cam.getSampleIndices().size() > 0) {  // use indices
    frames = frame_estimator.calculateLocalFrames(
        cloud_cam, cloud_cam.getSampleIndices(), params_.nn_radius_frames_,
        kdtree);
  } else {
    std::cout << "Error: No samples or no indices!\n";
    std::vector<std::unique_ptr<HandSet>> hand_set_list(0);
    // hand_set_list.resize(0);
    return hand_set_list;
  }

  if (plots_local_axes_) {
    plot_->plotLocalAxes(frames, cloud_cam.getCloudOriginal());
  }

  // 2. Evaluate possible hand placements.
  std::cout << "Finding hand poses ...\n";
  std::vector<std::unique_ptr<HandSet>> hand_set_list =
      evalHands(cloud_cam, frames, kdtree);

  const double t2 = omp_get_wtime();
  std::cout << "====> HAND SEARCH TIME: " << t2 - t0_total << std::endl;

  return hand_set_list;
}

std::vector<int> HandSearch::reevaluateHypotheses(
    const util::Cloud &cloud_cam,
    std::vector<std::unique_ptr<candidate::Hand>> &grasps,
    bool plot_samples) const {
  // Create KdTree for neighborhood search.
  const Eigen::MatrixXi &camera_source = cloud_cam.getCameraSource();
  const Eigen::Matrix3Xd &cloud_normals = cloud_cam.getNormals();
  const PointCloudRGB::Ptr &cloud = cloud_cam.getCloudProcessed();
  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud);

  if (plot_samples) {
    Eigen::Matrix3Xd samples(3, grasps.size());
    for (int i = 0; i < grasps.size(); i++) {
      samples.col(i) = grasps[i]->getSample();
    }

    plot_->plotSamples(samples, cloud);
  }

  std::vector<int> nn_indices;
  std::vector<float> nn_dists;
  Eigen::Matrix3Xd points =
      cloud->getMatrixXfMap().block(0, 0, 3, cloud->size()).cast<double>();
  util::PointList point_list(points, cloud_normals, camera_source,
                             cloud_cam.getViewPoints());
  util::PointList nn_points;
  std::vector<int> labels(grasps.size());

#ifdef _OPENMP
#pragma omp parallel for private(nn_indices, nn_dists, nn_points) \
    num_threads(params_.num_threads_)
#endif
  for (int i = 0; i < grasps.size(); i++) {
    labels[i] = 0;
    grasps[i]->setHalfAntipodal(false);
    grasps[i]->setFullAntipodal(false);
    const Eigen::Vector3d &sample = grasps[i]->getSample();
    pcl::PointXYZRGBA sample_pcl = eigenVectorToPcl(sample);

    if (kdtree.radiusSearch(sample_pcl, nn_radius_, nn_indices, nn_dists) > 0) {
      nn_points = point_list.slice(nn_indices);
      util::PointList nn_points_frame;
      FingerHand finger_hand(params_.hand_geometry_.finger_width_,
                             params_.hand_geometry_.outer_diameter_,
                             params_.hand_geometry_.depth_,
                             params_.num_finger_placements_);

      // Set the lateral and forward axes of the robot hand frame (closing
      // direction and grasp approach direction).
      finger_hand.setForwardAxis(0);
      finger_hand.setLateralAxis(1);

      // Check for collisions and if the hand contains at least one point.
      if (reevaluateHypothesis(nn_points, *grasps[i], finger_hand,
                               nn_points_frame)) {
        int label = labelHypothesis(nn_points_frame, finger_hand);
        if (label == Antipodal::FULL_GRASP) {
          labels[i] = 1;
          grasps[i]->setFullAntipodal(true);
        } else if (label == Antipodal::HALF_GRASP) {
          grasps[i]->setHalfAntipodal(true);
        }
      }
    }
  }

  return labels;
}

pcl::PointXYZRGBA HandSearch::eigenVectorToPcl(const Eigen::Vector3d &v) const {
  pcl::PointXYZRGBA p;
  p.x = v(0);
  p.y = v(1);
  p.z = v(2);
  return p;
}

std::vector<std::unique_ptr<candidate::HandSet>> HandSearch::evalHands(
    const util::Cloud &cloud_cam,
    const std::vector<candidate::LocalFrame> &frames,
    const pcl::KdTreeFLANN<pcl::PointXYZRGBA> &kdtree) const {
  double t1 = omp_get_wtime();

  // possible angles used for hand orientations
  const Eigen::VectorXd angles_space = Eigen::VectorXd::LinSpaced(
      params_.num_orientations_ + 1, -1.0 * M_PI / 2.0, M_PI / 2.0);

  // necessary b/c assignment in Eigen does not change vector size
  const Eigen::VectorXd angles = angles_space.head(params_.num_orientations_);

  std::vector<int> nn_indices;
  std::vector<float> nn_dists;
  const PointCloudRGB::Ptr &cloud = cloud_cam.getCloudProcessed();
  const Eigen::Matrix3Xd points =
      cloud->getMatrixXfMap().block(0, 0, 3, cloud->size()).cast<double>();
  std::vector<std::unique_ptr<HandSet>> hand_set_list(frames.size());
  const util::PointList point_list(points, cloud_cam.getNormals(),
                                   cloud_cam.getCameraSource(),
                                   cloud_cam.getViewPoints());
  util::PointList nn_points;

#ifdef _OPENMP  // parallelization using OpenMP
#pragma omp parallel for private(nn_indices, nn_dists, nn_points) \
    num_threads(params_.num_threads_)
#endif
  for (std::size_t i = 0; i < frames.size(); i++) {
    pcl::PointXYZRGBA sample = eigenVectorToPcl(frames[i].getSample());
    hand_set_list[i] = std::make_unique<HandSet>(
        params_.hand_geometry_, angles, params_.hand_axes_,
        params_.num_finger_placements_, params_.deepen_hand_, *antipodal_);

    if (kdtree.radiusSearch(sample, nn_radius_, nn_indices, nn_dists) > 0) {
      nn_points = point_list.slice(nn_indices);
      hand_set_list[i]->evalHandSet(nn_points, frames[i]);
    }
  }

  printf("Found %d hand sets in %3.2fs\n", (int)hand_set_list.size(),
         omp_get_wtime() - t1);

  return hand_set_list;
}

bool HandSearch::reevaluateHypothesis(
    const util::PointList &point_list, const candidate::Hand &hand,
    FingerHand &finger_hand, util::PointList &point_list_cropped) const {
  // Transform points into hand frame and crop them on <hand_height>.
  util::PointList point_list_frame = point_list.transformToHandFrame(
      hand.getSample(), hand.getFrame().transpose());
  point_list_cropped =
      point_list_frame.cropByHandHeight(params_.hand_geometry_.height_);

  // Check that the finger placement is possible.
  finger_hand.evaluateFingers(point_list_cropped.getPoints(), hand.getTop(),
                              hand.getFingerPlacementIndex());
  finger_hand.evaluateHand(hand.getFingerPlacementIndex());

  if (finger_hand.getHand().any()) {
    return true;
  }

  return false;
}

int HandSearch::labelHypothesis(const util::PointList &point_list,
                                FingerHand &finger_hand) const {
  std::vector<int> indices_learning =
      finger_hand.computePointsInClosingRegion(point_list.getPoints());
  if (indices_learning.size() == 0) {
    return Antipodal::NO_GRASP;
  }

  // extract data for classification
  util::PointList point_list_learning = point_list.slice(indices_learning);

  // evaluate if the grasp is antipodal
  int antipodal_result = antipodal_->evaluateGrasp(
      point_list_learning, 0.003, finger_hand.getLateralAxis(),
      finger_hand.getForwardAxis(), 2);

  return antipodal_result;
}

}  // namespace candidate
}  // namespace gpd
