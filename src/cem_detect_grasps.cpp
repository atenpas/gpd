#include <string>

#include <gpd/sequential_importance_sampling.h>

namespace gpd {
namespace apps {
namespace detect_grasps {

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

int DoMain(int argc, char *argv[]) {
  // Read arguments from command line.
  if (argc < 3) {
    std::cout << "Error: Not enough input arguments!\n\n";
    std::cout
        << "Usage: cem_detect_grasps CONFIG_FILE PCD_FILE [NORMALS_FILE]\n\n";
    std::cout << "Detect grasp poses for a point cloud, PCD_FILE (*.pcd), "
                 "using parameters from CONFIG_FILE (*.cfg).\n\n";
    std::cout << "[NORMALS_FILE] (optional) contains a surface normal for each "
                 "point in the cloud (*.csv).\n";
    return (-1);
  }

  std::string config_filename = argv[1];
  std::string pcd_filename = argv[2];
  if (!checkFileExists(config_filename)) {
    printf("Error: config file not found!\n");
    return (-1);
  }
  if (!checkFileExists(pcd_filename)) {
    printf("Error: PCD file not found!\n");
    return (-1);
  }

  // View point from which the camera sees the point cloud.
  Eigen::Matrix3Xd view_points(3, 1);
  view_points.setZero();

  // Load point cloud from file
  util::Cloud cloud(pcd_filename, view_points);
  if (cloud.getCloudOriginal()->size() == 0) {
    std::cout << "Error: Input point cloud is empty or does not exist!\n";
    return (-1);
  }

  // Load surface normals from file.
  if (argc > 3) {
    std::string normals_filename = argv[3];
    cloud.setNormalsFromFile(normals_filename);
    std::cout << "Loaded surface normals from file: " << normals_filename
              << "\n";
  }

  // Read parameters from configuration file.
  const double VOXEL_SIZE = 0.003;
  util::ConfigFile config_file(config_filename);
  config_file.ExtractKeys();
  std::vector<double> workspace =
      config_file.getValueOfKeyAsStdVectorDouble("workspace", "-1 1 -1 1 -1 1");
  int num_threads = config_file.getValueOfKey<int>("num_threads", 1);
  bool sample_above_plane =
      config_file.getValueOfKey<int>("sample_above_plane", false);
  double normals_radius =
      config_file.getValueOfKey<double>("normals_radius", 0.03);
  printf("num_threads: %d\n", num_threads);
  printf("sample_above_plane: %d\n", sample_above_plane);
  printf("normals_radius: %.3f\n", normals_radius);

  // Preprocess the point cloud.
  cloud.filterWorkspace(workspace);
  cloud.voxelizeCloud(VOXEL_SIZE);
  cloud.calculateNormals(num_threads, normals_radius);
  if (sample_above_plane) {
    cloud.sampleAbovePlane();
  }

  // Detect grasp affordances.
  SequentialImportanceSampling detector(config_filename);
  detector.detectGrasps(cloud);

  return 0;
}

}  // namespace detect_grasps
}  // namespace apps
}  // namespace gpd

int main(int argc, char *argv[]) {
  return gpd::apps::detect_grasps::DoMain(argc, argv);
}
