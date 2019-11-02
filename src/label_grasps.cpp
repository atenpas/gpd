#include <string>

#include <gpd/grasp_detector.h>

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
  if (argc < 4) {
    std::cout << "Error: Not enough input arguments!\n\n";
    std::cout << "Usage: label_grasps CONFIG_FILE PCD_FILE MESH_FILE\n\n";
    std::cout << "Find grasp poses for a point cloud, PCD_FILE (*.pcd), "
                 "using parameters from CONFIG_FILE (*.cfg), and check them "
                 "against a mesh, MESH_FILE (*.pcd).\n\n";
    return (-1);
  }

  std::string config_filename = argv[1];
  std::string pcd_filename = argv[2];
  std::string mesh_filename = argv[3];
  if (!checkFileExists(config_filename)) {
    printf("Error: CONFIG_FILE not found!\n");
    return (-1);
  }
  if (!checkFileExists(pcd_filename)) {
    printf("Error: PCD_FILE not found!\n");
    return (-1);
  }
  if (!checkFileExists(mesh_filename)) {
    printf("Error: MESH_FILE not found!\n");
    return (-1);
  }

  // Read parameters from configuration file.
  const double VOXEL_SIZE = 0.003;
  util::ConfigFile config_file(config_filename);
  config_file.ExtractKeys();
  std::vector<double> workspace =
      config_file.getValueOfKeyAsStdVectorDouble("workspace", "-1 1 -1 1 -1 1");
  int num_threads = config_file.getValueOfKey<int>("num_threads", 1);
  int num_samples = config_file.getValueOfKey<int>("num_samples", 100);
  bool sample_above_plane =
      config_file.getValueOfKey<int>("sample_above_plane", 1);
  double normals_radius =
      config_file.getValueOfKey<double>("normals_radius", 0.03);
  printf("num_threads: %d, num_samples: %d\n", num_threads, num_samples);
  printf("sample_above_plane: %d\n", sample_above_plane);
  printf("normals_radius: %.3f\n", normals_radius);

  // View point from which the camera sees the point cloud.
  Eigen::Matrix3Xd view_points(3, 1);
  view_points.setZero();

  // Load point cloud from file.
  util::Cloud cloud(pcd_filename, view_points);
  if (cloud.getCloudOriginal()->size() == 0) {
    std::cout << "Error: Input point cloud is empty or does not exist!\n";
    return (-1);
  }

  // Load point cloud from file.
  util::Cloud mesh(mesh_filename, view_points);
  if (mesh.getCloudOriginal()->size() == 0) {
    std::cout << "Error: Mesh point cloud is empty or does not exist!\n";
    return (-1);
  }

  // Prepare the point cloud.
  cloud.filterWorkspace(workspace);
  cloud.voxelizeCloud(VOXEL_SIZE);
  cloud.calculateNormals(num_threads, normals_radius);
  cloud.setNormals(cloud.getNormals() * (-1.0));  // TODO: do not do this!
  if (sample_above_plane) {
    cloud.sampleAbovePlane();
  }
  cloud.subsample(num_samples);

  // Prepare the mesh.
  mesh.calculateNormals(num_threads, normals_radius);
  mesh.setNormals(mesh.getNormals() * (-1.0));

  // Detect grasp poses.
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> images;
  GraspDetector detector(config_filename);
  detector.createGraspImages(cloud, hands, images);

  std::vector<int> labels = detector.evalGroundTruth(mesh, hands);
  printf("labels: %zu\n", labels.size());

  const candidate::HandSearch::Parameters &params =
      detector.getHandSearchParameters();
  util::Plot plot(params.hand_axes_.size(), params.num_orientations_);
  plot.plotAntipodalHands(hands, cloud.getCloudProcessed(), "Labeled Hands",
                          params.hand_geometry_);

  std::vector<std::unique_ptr<candidate::Hand>> valid_hands;
  for (size_t i = 0; i < hands.size(); i++) {
    printf("(%zu) label: %d\n", i, labels[i]);
    if (hands[i]->isFullAntipodal()) {
      printf("(%zu) label: %d\n", i, labels[i]);
      valid_hands.push_back(std::move(hands[i]));
    }
  }
  plot.plotValidHands(valid_hands, cloud.getCloudProcessed(),
                      mesh.getCloudProcessed(), "Antipodal Hands",
                      params.hand_geometry_);
  // plot.plotFingers3D(valid_hands, cloud.getCloudProcessed(), "Antipodal
  // Hands",
  //                    params.hand_geometry_);

  return 0;
}

}  // namespace detect_grasps
}  // namespace apps
}  // namespace gpd

int main(int argc, char *argv[]) {
  return gpd::apps::detect_grasps::DoMain(argc, argv);
}
