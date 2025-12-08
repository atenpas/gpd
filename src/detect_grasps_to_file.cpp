#include <string>
#include <gpd/grasp_detector.h>
#include <fstream>
#include <iostream>
#include <iomanip>

bool checkFileExists(const std::string &file_name) {
  std::ifstream file;
  file.open(file_name.c_str());
  if (!file) {
    std::cout << "File " + file_name + " could not be found!\n";
    return false;
  }
  file.close();
  return true;
}

int main(int argc, char* argv[])
{
  if (argc < 4)
  {
    std::cout << "Usage: detect_grasps_to_file CONFIG_FILE PCD_FILE OUTPUT_FILE\n";
    std::cout << "\nDetect grasp poses and save to a structured text file.\n";
    return -1;
  }

  std::string config_filename = argv[1];
  std::string pcd_filename = argv[2];
  std::string output_filename = argv[3];

  if (!checkFileExists(config_filename)) {
    std::cout << "Error: config file not found!\n";
    return -1;
  }
  if (!checkFileExists(pcd_filename)) {
    std::cout << "Error: PCD file not found!\n";
    return -1;
  }

  std::cout << "=== GPD Grasp Detection ===" << std::endl;
  std::cout << "Config file: " << config_filename << std::endl;
  std::cout << "Input PCD:   " << pcd_filename << std::endl;
  std::cout << "Output file: " << output_filename << std::endl;
  std::cout << std::endl;

  try
  {
    // Read parameters from configuration file
    std::cout << "Initializing GPD detector..." << std::endl;
    gpd::util::ConfigFile config_file(config_filename);
    config_file.ExtractKeys();
    
    // Set camera position from config (same as detect_grasps.cpp)
    std::vector<double> camera_position =
        config_file.getValueOfKeyAsStdVectorDouble("camera_position", "0.0 0.0 0.0");
    
    Eigen::Matrix3Xd view_points(3, 1);
    view_points << camera_position[0], camera_position[1], camera_position[2];
    
    // Load point cloud with camera view points
    std::cout << "Loading point cloud..." << std::endl;
    gpd::util::Cloud cloud(pcd_filename, view_points);
    
    if (cloud.getCloudOriginal()->size() == 0)
    {
      std::cerr << "Error: Input point cloud is empty or does not exist!" << std::endl;
      return -1;
    }
    
    std::cout << "Loaded " << cloud.getCloudOriginal()->points.size() << " points" << std::endl;
    
    // Create detector
    gpd::GraspDetector detector(config_filename);
    
    // CRITICAL: Preprocess the point cloud (voxelize, calculate normals, etc.)
    std::cout << "Preprocessing point cloud..." << std::endl;
    detector.preprocessPointCloud(cloud);

    // Detect grasps
    std::cout << "Detecting grasps (this may take a while)..." << std::endl;
    std::vector<std::unique_ptr<gpd::candidate::Hand>> grasps = detector.detectGrasps(cloud);
    
    std::cout << "Detected " << grasps.size() << " grasps" << std::endl;

    // Save grasps to file
    std::cout << "Saving grasps to " << output_filename << "..." << std::endl;
    std::ofstream outfile(output_filename);
    
    if (!outfile.is_open())
    {
      std::cerr << "Error: Cannot open output file: " << output_filename << std::endl;
      return -1;
    }

    outfile << "# GPD Detected Grasps\n";
    outfile << "# Generated from: " << pcd_filename << "\n";
    outfile << "# Format: x y z qw qx qy qz score width\n";
    outfile << grasps.size() << "\n";
    
    for (size_t i = 0; i < grasps.size(); i++)
    {
      const auto& grasp = grasps[i];
      Eigen::Vector3d position = grasp->getPosition();
      Eigen::Matrix3d rotation = grasp->getFrame();
      Eigen::Quaterniond quat(rotation);
      quat.normalize();
      
      double score = grasp->getScore();
      double width = grasp->getGraspWidth();
      
      outfile << std::fixed << std::setprecision(6)
              << position.x() << " " 
              << position.y() << " " 
              << position.z() << " "
              << quat.w() << " "
              << quat.x() << " "
              << quat.y() << " "
              << quat.z() << " "
              << score << " "
              << width << "\n";
    }
    
    outfile.close();
    std::cout << "\nSuccessfully saved " << grasps.size() << " grasps to " 
              << output_filename << std::endl;

    // Print summary
    if (grasps.size() > 0)
    {
      std::cout << "\n=== Top 3 Grasps ===" << std::endl;
      for (int i = 0; i < std::min(3, (int)grasps.size()); i++)
      {
        Eigen::Vector3d pos = grasps[i]->getPosition();
        std::cout << "  Grasp " << (i+1) << ": "
                  << "score=" << std::fixed << std::setprecision(3) << grasps[i]->getScore() 
                  << ", pos=(" << pos.x() << "," << pos.y() << "," << pos.z() << ")"
                  << ", width=" << grasps[i]->getGraspWidth() << "m" << std::endl;
      }
    }
    
    std::cout << "\nDone!" << std::endl;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}