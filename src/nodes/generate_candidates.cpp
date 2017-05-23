// ROS
#include <ros/ros.h>

// Custom
#include <gpg/cloud_camera.h>
#include <gpg/candidates_generator.h>


int main(int argc, char* argv[]) 
{
  // initialize ROS
  ros::init(argc, argv, "generate_grasp_candidates");
  ros::NodeHandle node("~");

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
  std::vector<double> camera_position;
  node.getParam("camera_position", camera_position);
  
  // Set the position from which the camera sees the point cloud.
  Eigen::Matrix3Xd view_points(3,1);
  view_points << camera_position[0], camera_position[1], camera_position[2];

  // Load point cloud from file
  std::string filename;
  node.param("cloud_file_name", filename, std::string(""));
  CloudCamera cloud_cam(filename, view_points);
  if (cloud_cam.getCloudOriginal()->size() == 0)
  {
    std::cout << "Input point cloud is empty or does not exist!\n";
    return (-1);
  }

  // Create object to generate grasp candidates.
  CandidatesGenerator candidates_generator(generator_params, hand_search_params);
  
  // Preprocess the point cloud: voxelization, removing statistical outliers, workspace filtering.
  candidates_generator.preprocessPointCloud(cloud_cam);
  
  // Generate grasp candidates.
  std::vector<Grasp> candidates = candidates_generator.generateGraspCandidates(cloud_cam);
  
  return 0;
}
