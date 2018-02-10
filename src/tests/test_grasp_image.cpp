#include <boost/lexical_cast.hpp>

#include <gpg/candidates_generator.h>
#include <gpg/cloud_camera.h>
#include <gpg/grasp.h>
#include <gpg/plot.h>

#include <gpd/learning.h>
#include <gpd/lenet.h>


int main(int argc, char* argv[])
{
  if (argc < 5)
  {
    std::cout << "Error: No input point cloud file given!\n";
    std::cout << "Usage: rosrun gpd test_grasp_image INPUT_FILE SAMPLE_INDEX DRAW_GRASP_IMAGES LENET_PARAMS_DIR\n";
    return -1;
  }

  // View point from which the camera sees the point cloud.
  Eigen::Matrix3Xd view_points(3,1);
  view_points.setZero();

  // Load point cloud from file
  std::string filename = argv[1];
  CloudCamera cloud_cam(filename, view_points);
  if (cloud_cam.getCloudOriginal()->size() == 0)
  {
    std::cout << "Error: Input point cloud is empty or does not exist!\n";
    return (-1);
  }

  // Use a custom sample.
//  Eigen::Matrix3Xd samples(3,1);
//  samples << -0.0129, 0.0515, 0.7042;
//  cloud_cam.setSamples(samples);

  // Read the sample index from the terminal.
  int sample_idx = boost::lexical_cast<int>(argv[2]);
  if (sample_idx >= cloud_cam.getCloudOriginal()->size())
  {
    std::cout << "Error: Sample index is larger than the number of points in the cloud!\n";
    return -1;
  }
  std::vector<int> sample_indices;
  sample_indices.push_back(sample_idx);
  cloud_cam.setSampleIndices(sample_indices);

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
  image_params.outer_diameter_ = 0.12;
  image_params.size_ = 60;
  image_params.num_channels_ = 15;

  // Calculate surface normals.
  cloud_cam.calculateNormals(4);
  cloud_cam.setNormals(cloud_cam.getNormals()*(-1.0)); // trick for GT

  // Plot the normals.
  Plot plot;
  plot.plotNormals(cloud_cam.getCloudProcessed(), cloud_cam.getNormals());

  // Generate grasp candidates.
  CandidatesGenerator candidates_generator(generator_params, hand_search_params);
  std::vector<GraspSet> hand_set_list = candidates_generator.generateGraspCandidateSets(cloud_cam);
  std::cout << hand_set_list[0].getHypotheses()[0].getSample().transpose() << std::endl;
  std::cout << hand_set_list[0].getHypotheses()[0].getFrame() << std::endl;
  std::cout << "bottom: " << hand_set_list[0].getHypotheses()[0].getGraspBottom().transpose() << std::endl;

  // Create the image for this grasp candidate.
  bool plot_images = true;
  plot_images = boost::lexical_cast<bool>(argv[3]);
  Learning learn(image_params, 1, hand_search_params.num_orientations_, plot_images, false);
  std::vector<cv::Mat> images = learn.createImages(cloud_cam, hand_set_list);

  // Evaluate if the grasp candidates are antipodal.
  std::cout << "Antipodal: ";
  for (int i=0; i < hand_set_list[0].getHypotheses().size(); i++)
  {
    std::cout << hand_set_list[0].getHypotheses()[i].isFullAntipodal() << " ";
  }
  std::cout << "\n";

  // Classify the grasp images.
  std::string lenet_params_dir = argv[4];
  Lenet net(1, lenet_params_dir);
  std::vector<float> scores2 = net.classifyImages(images);

  std::cout << "Scores: ";
  for (int i=0; i < scores2.size(); i++)
  {
    std::cout << scores2[i] << " ";
  }
  std::cout << "\n";

  return 0;
}
