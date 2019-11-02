#include <stdio.h>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <gpd/util/cloud.h>

#include <gpd/data_generator.h>
#include <gpd/grasp_detector.h>

namespace gpd {
namespace detect_grasps_python {

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudRGB;

struct Position {
  double x, y, z;

  //  Position();
  //
  //  Position(double x, double y, double z) : px(x), py(y), pz(z) { }
  //
  //  Position(const Eigen::Vector3d& v) : px(v(0)), py(v(1)), pz(v(2)) { }
};

struct Quaternion {
  double x, y, z, w;

  //  Quaternion();
  //
  //  Quaternion(double x, double y, double z, double w) : qx(x), qy(y), qz(z),
  //  qw(w) { }
  //
  //  Quaternion(const Eigen::Quaterniond& v) : qx(v.x()), qy(v.y()), qz(v.z()),
  //  qw(v.w()) { }
};

// struct Grasp
//{
//  Position pos;
//  Quaternion orient;
//  Position sample;
//  double score;
//  bool label;
//};

extern "C" struct Grasp {
  double *pos;
  double *orient;
  double *sample;
  double score;
  bool label;
  int *image;
};

extern "C" struct AugmentedCloud {
  float *points;
  float *normals;
  int *camera_index;
  float *view_points;
  int size;
  int num_view_points;
};

Eigen::Matrix3Xd viewPointsToMatrix(float *array, int n) {
  Eigen::Matrix3Xd mat(3, n);

  for (int i = 0; i < n; i++) {
    mat.col(i) << array[3 * i], array[3 * i + 1], array[3 * i + 2];
  }

  return mat;
}

Eigen::MatrixXi cameraSourceToMatrix(int *array, int rows, int cols) {
  Eigen::MatrixXi mat(rows, cols);

  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {
      mat(j, i) = array[i * rows + j];
    }
  }

  return mat;
}

int *cvMatToArray(const cv::Mat &image) {
  printf("HELLO\n");
  printf("%d, %d, %d\n", image.rows, image.rows, image.channels());
  int *array = new int[image.rows * image.cols * image.channels()];
  int l = 0;
  printf("%d, %d, %d\n", image.rows, image.rows, image.channels());

  for (int i = 0; i < image.rows; i++) {
    const uchar *ptr = image.ptr(i);

    for (int j = 0; j < image.cols; j++) {
      const uchar *uc_pixel = ptr;

      for (int k = 0; k < image.channels(); k++) {
        if ((int)uc_pixel[k] > 0) {
          printf("%d, %d, %d\n", i, j, k);
          std::cout << "uc_pixel:" << (int)uc_pixel[k] << "\n";
        }
        //        array[l] = uc_pixel[k];
        l++;
        //        printf(" %d\n", uc_pixel[k]);
        //        std::cout << image.at<uchar>(i, j, k) << "\n";
        //        image.at<uchar>(i, j, k)
        //        array[l] = (float) image.at<uchar>(i, j, k);
        //        printf("%.2f\n", array[l]);
        l++;
      }
    }
  }

  return array;
}

PointCloudRGB arrayToPCLPointCloud(float *points, int num_points) {
  PointCloudRGB cloud;
  cloud.resize(num_points);

  for (int i = 0; i < num_points; i++) {
    PointT p;
    p.x = points[3 * i + 0];
    p.y = points[3 * i + 1];
    p.z = points[3 * i + 2];
    cloud.at(i) = p;
  }

  return cloud;
}

Cloud augmentedCloudToCloud(AugmentedCloud *cloud) {
  PointCloudRGB::Ptr pcl_cloud(new PointCloudRGB);
  *pcl_cloud = arrayToPCLPointCloud(cloud->points, cloud->size);
  Eigen::MatrixXi camera_source = cameraSourceToMatrix(
      cloud->camera_index, cloud->num_view_points, cloud->size);
  Eigen::Matrix3Xd view_points =
      viewPointsToMatrix(cloud->view_points, cloud->num_view_points);
  Eigen::Matrix3Xd normals = viewPointsToMatrix(cloud->normals, cloud->size);

  Cloud cloud_cam(pcl_cloud, camera_source, view_points);
  cloud_cam.setNormals(normals);
  return cloud_cam;
}

Cloud createCloud(float *points, int *camera_index, float *view_points,
                  int size, int num_view_points) {
  PointCloudRGB::Ptr pcl_cloud(new PointCloudRGB);
  *pcl_cloud = arrayToPCLPointCloud(points, size);
  Eigen::MatrixXi camera_index_mat =
      cameraSourceToMatrix(camera_index, num_view_points, size);
  Eigen::Matrix3Xd view_points_mat =
      viewPointsToMatrix(view_points, num_view_points);

  Cloud cloud_cam(pcl_cloud, camera_index_mat, view_points_mat);

  return cloud_cam;
}

Cloud createCloudNormals(float *points, float *normals, int *camera_index,
                         float *view_points, int size, int num_view_points) {
  PointCloudRGB::Ptr pcl_cloud(new PointCloudRGB);
  *pcl_cloud = arrayToPCLPointCloud(points, size);
  Eigen::Matrix3Xd normals_mat = viewPointsToMatrix(normals, size);
  Eigen::MatrixXi camera_index_mat =
      cameraSourceToMatrix(camera_index, num_view_points, size);
  Eigen::Matrix3Xd view_points_mat =
      viewPointsToMatrix(view_points, num_view_points);

  Cloud cloud_cam(pcl_cloud, camera_index_mat, view_points_mat);
  cloud_cam.setNormals(normals_mat);

  return cloud_cam;
}

Cloud createGroundTruthCloud(float *points, float *normals, int size) {
  PointCloudRGB::Ptr pcl_cloud(new PointCloudRGB);
  *pcl_cloud = arrayToPCLPointCloud(points, size);
  Eigen::Matrix3Xd points_gt = viewPointsToMatrix(points, size);
  Eigen::Matrix3Xd normals_gt = viewPointsToMatrix(normals, size);
  Eigen::MatrixXi cam_sources_gt = Eigen::MatrixXi::Ones(1, size);
  Eigen::Matrix3Xd cam_pos_gt = Eigen::Matrix3Xd::Zero(3, 1);

  Cloud mesh_cloud(pcl_cloud, cam_sources_gt, cam_pos_gt);
  mesh_cloud.setNormals(normals_gt);

  return mesh_cloud;
}

Position vectorToPositionStruct(const Eigen::Vector3d &v) {
  Position p;
  p.x = v(0);
  p.y = v(1);
  p.z = v(2);
  return p;
}

Quaternion matrixToQuaternionStruct(const Eigen::Matrix3d &rot) {
  Eigen::Quaterniond quat_eig(rot);
  Quaternion q;
  q.x = quat_eig.x();
  q.y = quat_eig.y();
  q.z = quat_eig.z();
  q.w = quat_eig.w();
  return q;
}

Cloud initCloud(char *pcd_filename, char *normals_filename, float *view_points,
                int num_view_points) {
  // Set view points from which the camera has taken the point cloud.
  Eigen::Matrix3Xd view_points_mat =
      viewPointsToMatrix(view_points, num_view_points);

  // Load point cloud from file.
  Cloud cloud(pcd_filename, view_points_mat);
  if (cloud.getCloudOriginal()->size() == 0) {
    printf("Error: Input point cloud is empty or does not exist!\n");
    return cloud;
  }

  // Optional: load surface normals from file.
  if (std::string(normals_filename).size() > 0) {
    cloud.setNormalsFromFile(normals_filename);
    printf("Loaded surface normals from file: %s\n", normals_filename);
  }

  return cloud;
}

// Grasp* handsToGraspsStruct(const std::vector<Hand>& hands)
//{
//  Grasp* grasps = new Grasp[hands.size()];
//
//  for (int i = 0; i < hands.size(); i++)
//  {
//    Position pos = vectorToPositionStruct(hands[i].getGraspBottom());
//    Quaternion quat = matrixToQuaternionStruct(hands[i].getFrame());
//    Position sample = vectorToPositionStruct(hands[i].getSample());
//
//    grasps[i].pos = pos;
//    grasps[i].orient = quat;
//    grasps[i].sample = sample;
//    grasps[i].score = hands[i].getScore();
//    grasps[i].label = hands[i].isFullAntipodal();
//  }
//
//  return grasps;
//}

Grasp *handsToGraspsStruct(const std::vector<Hand> &hands) {
  Grasp *grasps = new Grasp[hands.size()];

  for (int i = 0; i < hands.size(); i++) {
    grasps[i].pos = new double[3];
    Eigen::Map<Eigen::Vector3d>(grasps[i].pos) = hands[i].getGraspBottom();
    grasps[i].orient = new double[4];
    Eigen::Quaterniond q(hands[i].getFrame());
    Eigen::Map<Eigen::Quaterniond>(grasps[i].orient) = q;
    grasps[i].sample = new double[3];
    grasps[i].score = hands[i].getScore();
    grasps[i].label = hands[i].isFullAntipodal();
    grasps[i].image = new int[0];
    grasps[i].image[0] = -1;
  }

  return grasps;
}

Grasp *handsToGraspsStruct(const std::vector<Hand> &hands,
                           const std::vector<cv::Mat> &images) {
  Grasp *grasps = new Grasp[hands.size()];

  for (int i = 0; i < hands.size(); i++) {
    grasps[i].pos = new double[3];
    Eigen::Map<Eigen::Vector3d>(grasps[i].pos) = hands[i].getGraspBottom();
    grasps[i].orient = new double[4];
    Eigen::Quaterniond q(hands[i].getFrame());
    Eigen::Map<Eigen::Quaterniond>(grasps[i].orient) = q;
    grasps[i].sample = new double[3];
    grasps[i].score = hands[i].getScore();
    grasps[i].label = hands[i].isFullAntipodal();
    printf("i: %d\n", i);
    grasps[i].image = cvMatToArray(images[0]);  // new
    // int[images[i].rows*images[i].cols*images[i].channels()];
    //    int* img = cvMatToArray(images[0]);
    //    grasps[i].image = img;
    //    delete[] img;
  }

  return grasps;
}

std::vector<Hand> detectGrasps(const std::string &config_filename,
                               Cloud &cloud_cam) {
  // Preprocess the point cloud.
  GraspDetector detector(config_filename);
  detector.preprocessPointCloud(cloud_cam);

  // Detect grasps.
  std::vector<Hand> hands = detector.detectGrasps(cloud_cam);

  return hands;
}

std::vector<Hand> generateGraspCandidates(const std::string &config_filename,
                                          Cloud &cloud_cam) {
  // Preprocess the point cloud.
  GraspDetector detector(config_filename);
  detector.preprocessPointCloud(cloud_cam);

  // Generate grasp candidates.
  std::vector<HandSet> hand_sets = detector.generateGraspCandidates(cloud_cam);

  std::vector<Hand> hands;
  for (int i = 0; i < hand_sets.size(); i++) {
    for (int j = 0; j < hand_sets[i].getHands().size(); j++) {
      if (hand_sets[i].getIsValid()[j]) {
        hands.push_back(hand_sets[i].getHands()[j]);
      }
    }
  }

  return hands;
}

void copyImageMatrix(const cv::Mat &src, cv::Mat &dst, int idx_in) {
  for (int j = 0; j < 60; j++) {
    for (int k = 0; k < 60; k++) {
      for (int l = 0; l < 15; l++) {
        int idx_src[3] = {j, k, l};
        int idx_dst[4] = {idx_in, j, k, l};
        dst.at<uchar>(idx_dst) = src.at<uchar>(idx_src);
      }
    }
  }
}

void handToTransform4x4(const Hand &hand, cv::Mat &tf, int idx_in) {
  const Eigen::Matrix3d orient = hand.getFrame();

  // Copy the orientation.
  int idx[3] = {idx_in, 0, 0};

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      idx[1] = i;
      idx[2] = j;
      tf.at<float>(idx) = orient(i, j);
    }
  }

  // Copy the position.
  idx[2] = 3;

  for (int j = 0; j < 3; j++) {
    idx[1] = j;
    tf.at<float>(idx) = hand.getGraspBottom()(j);
  }
}

bool calcDescriptorsHelper(Cloud &cloud, GraspDetector &detector,
                           char *filename_out, int compress_level) {
  const std::string IMAGES_DS_NAME = "images";
  const std::string HANDS_DS_NAME = "hands";
  const std::string WIDTHS_DS_NAME = "widths";

  // Calculate grasps and grasp descriptors.
  std::vector<Hand> hands;
  std::vector<cv::Mat> images;
  bool found_hands = detector.createGraspImages(cloud, hands, images);

  if (not found_hands) {
    return false;
  }

  // Store the grasps and the grasp descriptors in a HDF5 file.
  int n_dims = 3;
  int dsdims_images[n_dims] = {images.size(), images[0].rows, images[0].cols};
  cv::Mat images_mat(n_dims, dsdims_images, CV_8UC(images[0].channels()),
                     cv::Scalar(0.0));

  int dsdims_hands[n_dims] = {hands.size(), 4, 4};
  cv::Mat hands_mat(n_dims, dsdims_hands, CV_32FC1, cv::Scalar(0.0));

  int n_dims_widths = 1;
  int dsdims_widths[n_dims_widths] = {hands.size()};
  cv::Mat gripper_widths_mat(n_dims_widths, dsdims_widths, CV_32FC1,
                             cv::Scalar(0.0));

  for (int i = 0; i < images.size(); i++) {
    copyImageMatrix(images[i], images_mat, i);
    handToTransform4x4(hands[i], hands_mat, i);
    gripper_widths_mat.at<float>(i) = hands[i].getGraspWidth();

    // ranges don't seem to work
    //    std::vector<cv::Range> ranges;
    //    ranges.push_back(cv::Range(i,i+1));
    //    ranges.push_back(cv::Range(0,60));
    //    ranges.push_back(cv::Range(0,60));
    //    data(&ranges[0]) = dataset[i].image_.clone();
    //    dataset[0].image_.copyTo(data(&ranges[0]));
    //    printf("dataset.label: %d,  labels(i): %d\n", dataset[i].label_,
    //    labels.at<uchar>(i));
  }

  cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open(filename_out);
  printf("Storing dataset %s ...\n", IMAGES_DS_NAME.c_str());
  h5io->dscreate(n_dims, dsdims_images, CV_8UC(images[0].channels()),
                 IMAGES_DS_NAME, compress_level);
  h5io->dswrite(images_mat, IMAGES_DS_NAME);

  printf("Storing dataset %s ...\n", HANDS_DS_NAME.c_str());
  h5io->dscreate(n_dims, dsdims_hands, CV_32FC1, HANDS_DS_NAME, compress_level);
  h5io->dswrite(hands_mat, HANDS_DS_NAME);

  printf("Storing dataset %s ...\n", WIDTHS_DS_NAME.c_str());
  h5io->dscreate(n_dims_widths, dsdims_widths, CV_32FC1, WIDTHS_DS_NAME,
                 compress_level);
  h5io->dswrite(gripper_widths_mat, WIDTHS_DS_NAME);

  h5io->close();

  return true;
}

extern "C" int detectGraspsInCloud(char *config_filename, float *points,
                                   int *camera_index, float *view_points,
                                   int size, int num_view_points,
                                   struct Grasp **grasps_out) {
  Cloud cloud =
      createCloud(points, camera_index, view_points, size, num_view_points);

  // Detect grasp affordances.
  std::string config_filename_str = config_filename;
  std::vector<Hand> hands = detectGrasps(config_filename_str, cloud);

  // Convert output to array of structs.
  *grasps_out = handsToGraspsStruct(hands);
  int num_out = (int)hands.size();

  return num_out;
}

extern "C" int detectGraspsInCloudNormals(char *config_filename, float *points,
                                          float *normals, int *camera_index,
                                          float *view_points, int size,
                                          int num_view_points,
                                          struct Grasp **grasps_out) {
  Cloud cloud = createCloudNormals(points, normals, camera_index, view_points,
                                   size, num_view_points);

  // Detect grasp affordances.
  std::string config_filename_str = config_filename;
  std::vector<Hand> hands = detectGrasps(config_filename_str, cloud);

  // Convert output to array of structs.
  *grasps_out = handsToGraspsStruct(hands);
  int num_out = (int)hands.size();

  return num_out;
}

extern "C" int detectGraspsInFile(char *config_filename, char *pcd_filename,
                                  char *normals_filename, float *view_points,
                                  int num_view_points,
                                  struct Grasp **grasps_out) {
  // Initialize cloud.
  Cloud cloud =
      initCloud(pcd_filename, normals_filename, view_points, num_view_points);
  if (cloud.getCloudOriginal()->size() == 0) {
    return 0;
  }

  // Detect grasp affordances.
  std::string config_filename_str = config_filename;
  std::vector<Hand> hands = detectGrasps(config_filename_str, cloud);

  // Convert output to array of structs.
  *grasps_out = handsToGraspsStruct(hands);
  int num_out = (int)hands.size();

  return num_out;
}

extern "C" int detectAndEvalGrasps(char *config_filename, float *points,
                                   int *camera_index, float *view_points,
                                   int size, int num_view_points,
                                   float *points_gt, float *normals_gt,
                                   int size_gt, struct Grasp **grasps_out) {
  // Create point cloud for current view.
  Cloud cloud =
      createCloud(points, camera_index, view_points, size, num_view_points);

  // Create point cloud for ground truth mesh.
  Cloud mesh_cloud = createGroundTruthCloud(points_gt, normals_gt, size_gt);

  // Preprocess the point cloud.
  GraspDetector detector(config_filename);
  detector.preprocessPointCloud(cloud);

  // Generate grasp candidates.
  std::vector<Hand> hands;
  std::vector<cv::Mat> images;
  bool found_hands = detector.createGraspImages(cloud, hands, images);
  if (found_hands == false) {
    printf("No grasps found!\n");
    return 0;
  }
  printf("Created %d grasps and %d images.\n", (int)hands.size(),
         (int)images.size());

  // Evaluate candidates against ground truth.
  std::vector<Hand> labeled_hands = detector.evalGroundTruth(mesh_cloud, hands);
  printf("Done!!\n");

  // Convert output to array of structs.
  *grasps_out = handsToGraspsStruct(labeled_hands, images);
  int num_out = (int)hands.size();

  printf("%d\n", num_out);

  return num_out;
}

extern "C" int generateGraspCandidatesInFile(
    char *config_filename, char *pcd_filename, char *normals_filename,
    float *view_points, int num_view_points, struct Grasp **grasps_out) {
  // Initialize point cloud.
  Cloud cloud =
      initCloud(pcd_filename, normals_filename, view_points, num_view_points);
  if (cloud.getCloudOriginal()->size() == 0) {
    return 0;
  }

  // Detect grasp affordances.
  std::string config_filename_str = config_filename;
  std::vector<Hand> hands = generateGraspCandidates(config_filename_str, cloud);

  // Convert output to array of structs.
  *grasps_out = handsToGraspsStruct(hands);
  int num_out = (int)hands.size();

  return num_out;
}

extern "C" int calcGraspDescriptorsAtIndices(
    char *config_filename, float *points, int *camera_index, float *view_points,
    int size, int num_view_points, int *indices, int num_indices,
    int compress_level, char *filename_out) {
  // Initialize point cloud.
  Cloud cloud =
      createCloud(points, camera_index, view_points, size, num_view_points);
  if (cloud.getCloudOriginal()->size() == 0) {
    return 0;
  }

  // Set indices at which to sample grasps.
  std::vector<int> sample_indices;
  sample_indices.resize(num_indices);
  for (int i = 0; i < num_indices; i++) {
    sample_indices[i] = indices[i];
  }
  cloud.setSampleIndices(sample_indices);

  // Calculate grasps and grasp descriptors.
  GraspDetector detector(config_filename);
  if (calcDescriptorsHelper(cloud, detector, filename_out, compress_level)) {
    return 1;
  }

  return 0;
}

extern "C" int calcGraspDescriptors(char *config_filename, float *points,
                                    int *camera_index, float *view_points,
                                    int size, int num_view_points,
                                    int compress_level, char *filename_out) {
  Cloud cloud =
      createCloud(points, camera_index, view_points, size, num_view_points);

  // Preprocess the point cloud.
  GraspDetector detector(config_filename);
  detector.preprocessPointCloud(cloud);

  // Calculate grasps and grasp descriptors.
  if (calcDescriptorsHelper(cloud, detector, filename_out, compress_level)) {
    return 1;
  }

  return 0;
}

extern "C" int freeMemoryGrasps(Grasp *in) {
  delete[] in;
  return 0;
}

extern "C" int CopyAndFree(float *in, float *out, int n) {
  memcpy(out, in, sizeof(float) * n);
  delete[] in;
  return 0;
}

}  // namespace detect_grasps_python
}  // namespace gpd
