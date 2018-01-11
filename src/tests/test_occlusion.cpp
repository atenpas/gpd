#include <gpg/candidates_generator.h>
#include <gpg/cloud_camera.h>
#include <gpg/grasp.h>
#include <gpg/plot.h>

#include <gpd/learning.h>


int main(int argc, char* argv[])
{
  // View point from which the camera sees the point cloud.
  Eigen::Matrix3Xd view_points(3,1);
  view_points.setZero();

  // Load point cloud from file
  //  std::string filename = "/media/andreas/2a9b7d00-f8c3-4849-9ddc-283f5b7c206a/data/object_datasets/bb_onesource/pcd/vo5_tea_therapy_healthful_green_tea_smoothing_shampoo_1_bin.pcd";
   std::string filename = "/home/andreas/data/bigbird/3m_high_track_spray_adhesive/clouds/NP1_0.pcd";
  // std::string filename = "/home/baxter/data/bigbird/3m_high_tack_spray_adhesive/clouds/NP1_0.pcd";
  CloudCamera cloud_cam(filename, view_points);
  if (cloud_cam.getCloudOriginal()->size() == 0)
  {
    std::cout << "Input point cloud is empty or does not exist!\n";
    return (-1);
  }

  // Use a custom sample.
  Eigen::Matrix3Xd samples(3,1);
  samples << -0.0129, 0.0515, 0.7042;
  cloud_cam.setSamples(samples);

  // Create objects to store parameters.
  CandidatesGenerator::Parameters generator_params;
  HandSearch::Parameters hand_search_params;

  // Hand geometry parameters
  hand_search_params.finger_width_ = 0.01;
  hand_search_params.hand_outer_diameter_ = 0.12;
  hand_search_params.hand_depth_ = 0.06;
  hand_search_params.hand_height_ = 0.02;
  hand_search_params.init_bite_ = 0.015;

  // Local hand search parameters
  hand_search_params.nn_radius_frames_ = 0.01;
  hand_search_params.num_orientations_ = 8;
  hand_search_params.num_samples_ = 1;
  hand_search_params.num_threads_ = 1;
  hand_search_params.rotation_axis_ = 2;

  // General parameters
  generator_params.num_samples_ = hand_search_params.num_samples_;
  generator_params.num_threads_ = hand_search_params.num_threads_;
  generator_params.plot_grasps_ = true;

  // Preprocessing parameters
  generator_params.remove_statistical_outliers_ = false;
  generator_params.voxelize_ = false;
  generator_params.workspace_.resize(6);
  generator_params.workspace_[0] = -1.0;
  generator_params.workspace_[1] = 1.0;
  generator_params.workspace_[2] = -1.0;
  generator_params.workspace_[3] = 1.0;
  generator_params.workspace_[4] = -1.0;
  generator_params.workspace_[5] = 1.0;

  // Image parameters
  Learning::ImageParameters image_params;
  image_params.depth_ = 0.06;
  image_params.height_ = 0.02;
  image_params.outer_diameter_ = 0.10;
  image_params.size_ = 60;
  image_params.num_channels_ = 15;

  // Calculate surface normals.
  cloud_cam.calculateNormals(4);

  // Plot the normals.
  Plot plot;
  plot.plotNormals(cloud_cam.getCloudProcessed(), cloud_cam.getNormals());

  // Manually construct a grasp candidate.
  Eigen::Matrix3d frame;
  frame <<  0.9317,    0.3561,    0.0717,
      0.0968,   -0.0533,   -0.9939,
      -0.3501,    0.9329,   -0.0841;
  Eigen::Vector3d sample = samples.col(0);
  FingerHand fh(hand_search_params.finger_width_, hand_search_params.hand_outer_diameter_, hand_search_params.hand_depth_);
  fh.setBottom(-0.0150);
  fh.setTop(0.0450);
  fh.setSurface(-0.0180);
  fh.setLeft(-0.0511);
  fh.setRight(0.0489);
  fh.setCenter(-0.0011);

  Grasp hand(sample, frame, fh);
  hand.print();

  std::vector<Grasp> hands;
  hands.push_back(hand);

  GraspSet hand_set;
  hand_set.setHands(hands);
  hand_set.setSample(sample);

  std::vector<GraspSet> hand_set_list;
  hand_set_list.push_back(hand_set);

  // Create the image for this grasp candidate.
  Learning learn(image_params, 1, hand_search_params.num_orientations_, true, false);
  std::vector<cv::Mat> images = learn.createImages(cloud_cam, hand_set_list);
  std::cout << "------------\n" << images[0].rows << " x " << images[0].cols << " x " << images[0].channels() << "\n";

  // Predict if the grasp candidate is a good grasp or not.

  return 0;
}
