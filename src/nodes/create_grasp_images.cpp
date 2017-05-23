// OpenCV
#include <opencv2/opencv.hpp>

// ROS
#include <ros/ros.h>

// Custom
#include <gpg/cloud_camera.h>
#include <gpg/candidates_generator.h>

#include "../../include/gpd/learning.h"


CandidatesGenerator createCandidatesGenerator(ros::NodeHandle& node)
{
  // Create objects to store parameters
  CandidatesGenerator::Parameters generator_params;
  HandSearch::Parameters hand_search_params;

  // Read hand geometry parameters
  node.param("finger_width", hand_search_params.finger_width_, 0.01);
  node.param("hand_outer_diameter", hand_search_params.hand_outer_diameter_, 0.09);
  node.param("hand_depth", hand_search_params.hand_depth_, 0.06);
  node.param("hand_height", hand_search_params.hand_height_, 0.02);
  node.param("init_bite", hand_search_params.init_bite_, 0.015);

  // Read local hand search parameters
  node.param("nn_radius", hand_search_params.nn_radius_frames_, 0.01);
  node.param("num_orientations", hand_search_params.num_orientations_, 8);
  node.param("num_samples", hand_search_params.num_samples_, 500);
  node.param("num_threads", hand_search_params.num_threads_, 1);
  node.param("rotation_axis", hand_search_params.rotation_axis_, 2);

  // Read general parameters
  generator_params.num_samples_ = hand_search_params.num_samples_;
  generator_params.num_threads_ = hand_search_params.num_threads_;
  node.param("plot_candidates", generator_params.plot_grasps_, false);

  // Read preprocessing parameters
  node.param("remove_outliers", generator_params.remove_statistical_outliers_, true);
  node.param("voxelize", generator_params.voxelize_, true);
  node.getParam("workspace", generator_params.workspace_);

  // Create object to generate grasp candidates.
  return CandidatesGenerator(generator_params, hand_search_params);
}


CloudCamera loadCloudCameraFromFile(ros::NodeHandle& node)
{
  // Set the position from which the camera sees the point cloud.
  std::vector<double> camera_position;
  node.getParam("camera_position", camera_position);
  Eigen::Matrix3Xd view_points(3,1);
  view_points << camera_position[0], camera_position[1], camera_position[2];

  // Load the point cloud from the file.
  std::string filename;
  node.param("cloud_file_name", filename, std::string(""));

  return CloudCamera(filename, view_points);
}


Learning createLearning(ros::NodeHandle& node)
{
  // Read grasp image parameters.
  Learning::ImageParameters image_params;
  node.param("image_outer_diameter", image_params.outer_diameter_, 0.09);
  node.param("image_depth", image_params.depth_, 0.06);
  node.param("image_height", image_params.height_, 0.02);
  node.param("image_size", image_params.size_, 60);
  node.param("image_num_channels", image_params.num_channels_, 15);

  // Read learning parameters.
  bool remove_plane;
  int num_orientations, num_threads;
  node.param("remove_plane_before_image_calculation", remove_plane, false);
  node.param("num_orientations", num_orientations, 8);
  node.param("num_threads", num_threads, 1);

  // Create object to create grasp images from grasp candidates (used for classification).
  return Learning(image_params, num_threads, num_orientations, false, remove_plane);
}


void extractGraspsAndImages(const std::vector<GraspSet>& hand_set_list, const std::vector<cv::Mat>& images,
  std::vector<Grasp>& grasps_out, std::vector<cv::Mat>& images_out)
{
  grasps_out.resize(0);
  images_out.resize(0);
  int num_orientations = hand_set_list[0].getHypotheses().size();

  for (int i = 0; i < hand_set_list.size(); i++)
  {
    const std::vector<Grasp>& hands = hand_set_list[i].getHypotheses();

    for (int j = 0; j < hands.size(); j++)
    {
      if (hand_set_list[i].getIsValid()(j))
      {
        grasps_out.push_back(hands[j]);
        images_out.push_back(images[i * num_orientations + j]);
      }
    }
  }
}


int main(int argc, char** argv)
{
  // Seed the random number generator.
  std::srand(std::time(0));

  // Initialize ROS.
  ros::init(argc, argv, "create_grasp_images");
  ros::NodeHandle node("~");

  // Load point cloud from file.
  CloudCamera cloud_cam = loadCloudCameraFromFile(node);
  if (cloud_cam.getCloudOriginal()->size() == 0)
  {
    ROS_ERROR("Point cloud is empty!");
    return (-1);
  }

  // Create generator for grasp candidates.
  CandidatesGenerator candidates_generator = createCandidatesGenerator(node);

  // Preprocess the point cloud.
  candidates_generator.preprocessPointCloud(cloud_cam);

  // Generate grasp candidates.
  std::vector<GraspSet> candidates = candidates_generator.generateGraspCandidateSets(cloud_cam);

  if (candidates.size() == 0)
  {
    ROS_ERROR("No grasp candidates found!");
    return (-1);
  }

  // Create object to generate grasp images.
  Learning learning = createLearning(node);

  // Create the grasp images.
  std::vector<cv::Mat> image_list = learning.createImages(cloud_cam, candidates);
  std::vector<Grasp> valid_grasps;
  std::vector<cv::Mat> valid_images;
  extractGraspsAndImages(candidates, image_list, valid_grasps, valid_images);

  // Store the grasp images.
  std::string file_out = "/home/andreas/test.xml";
  cv::FileStorage file_storage(file_out, cv::FileStorage::WRITE);
  for (int i = 0; i < valid_images.size(); i++)
  {
    file_storage << "image_" + boost::lexical_cast<std::string>(i) << image_list[i];
    file_storage << "label_" + boost::lexical_cast<std::string>(i) << valid_grasps[i].isFullAntipodal();
  }
  file_storage.release();

  return 0;
}
