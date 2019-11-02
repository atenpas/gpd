#include <gpd/util/cloud.h>

#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace gpd {
namespace util {

Cloud::Cloud()
    : cloud_original_(new PointCloudRGB), cloud_processed_(new PointCloudRGB) {
  view_points_.resize(3, 1);
  view_points_ << 0.0, 0.0, 0.0;
  sample_indices_.resize(0);
  samples_.resize(3, 0);
  normals_.resize(3, 0);
}

Cloud::Cloud(const PointCloudRGB::Ptr &cloud,
             const Eigen::MatrixXi &camera_source,
             const Eigen::Matrix3Xd &view_points)
    : cloud_processed_(new PointCloudRGB),
      cloud_original_(new PointCloudRGB),
      camera_source_(camera_source),
      view_points_(view_points) {
  sample_indices_.resize(0);
  samples_.resize(3, 0);
  normals_.resize(3, 0);

  pcl::copyPointCloud(*cloud, *cloud_original_);
  *cloud_processed_ = *cloud_original_;
}

Cloud::Cloud(const PointCloudPointNormal::Ptr &cloud,
             const Eigen::MatrixXi &camera_source,
             const Eigen::Matrix3Xd &view_points)
    : cloud_processed_(new PointCloudRGB),
      cloud_original_(new PointCloudRGB),
      camera_source_(camera_source),
      view_points_(view_points) {
  sample_indices_.resize(0);
  samples_.resize(3, 0);
  normals_.resize(3, 0);

  pcl::copyPointCloud(*cloud, *cloud_original_);
  *cloud_processed_ = *cloud_original_;
}

Cloud::Cloud(const PointCloudPointNormal::Ptr &cloud, int size_left_cloud,
             const Eigen::Matrix3Xd &view_points)
    : cloud_processed_(new PointCloudRGB),
      cloud_original_(new PointCloudRGB),
      view_points_(view_points) {
  sample_indices_.resize(0);
  samples_.resize(3, 0);

  pcl::copyPointCloud(*cloud, *cloud_original_);
  *cloud_processed_ = *cloud_original_;

  // set the camera source matrix: (i,j) = 1 if point j is seen by camera i
  if (size_left_cloud == 0)  // one camera
  {
    camera_source_ = Eigen::MatrixXi::Ones(1, cloud->size());
  } else  // two cameras
  {
    int size_right_cloud = cloud->size() - size_left_cloud;
    camera_source_ = Eigen::MatrixXi::Zero(2, cloud->size());
    camera_source_.block(0, 0, 1, size_left_cloud) =
        Eigen::MatrixXi::Ones(1, size_left_cloud);
    camera_source_.block(1, size_left_cloud, 1, size_right_cloud) =
        Eigen::MatrixXi::Ones(1, size_right_cloud);
  }

  normals_.resize(3, cloud->size());
  for (int i = 0; i < cloud->size(); i++) {
    normals_.col(i) << cloud->points[i].normal_x, cloud->points[i].normal_y,
        cloud->points[i].normal_z;
  }
}

Cloud::Cloud(const PointCloudRGB::Ptr &cloud, int size_left_cloud,
             const Eigen::Matrix3Xd &view_points)
    : cloud_processed_(cloud),
      cloud_original_(cloud),
      view_points_(view_points) {
  sample_indices_.resize(0);
  samples_.resize(3, 0);
  normals_.resize(3, 0);

  // set the camera source matrix: (i,j) = 1 if point j is seen by camera i
  if (size_left_cloud == 0)  // one camera
  {
    camera_source_ = Eigen::MatrixXi::Ones(1, cloud->size());
  } else  // two cameras
  {
    int size_right_cloud = cloud->size() - size_left_cloud;
    camera_source_ = Eigen::MatrixXi::Zero(2, cloud->size());
    camera_source_.block(0, 0, 1, size_left_cloud) =
        Eigen::MatrixXi::Ones(1, size_left_cloud);
    camera_source_.block(1, size_left_cloud, 1, size_right_cloud) =
        Eigen::MatrixXi::Ones(1, size_right_cloud);
  }
}

Cloud::Cloud(const std::string &filename, const Eigen::Matrix3Xd &view_points)
    : cloud_processed_(new PointCloudRGB),
      cloud_original_(new PointCloudRGB),
      view_points_(view_points) {
  sample_indices_.resize(0);
  samples_.resize(3, 0);
  normals_.resize(3, 0);
  cloud_processed_ = loadPointCloudFromFile(filename);
  cloud_original_ = cloud_processed_;
  camera_source_ = Eigen::MatrixXi::Ones(1, cloud_processed_->size());
  std::cout << "Loaded point cloud with " << camera_source_.cols()
            << " points \n";
}

Cloud::Cloud(const std::string &filename_left,
             const std::string &filename_right,
             const Eigen::Matrix3Xd &view_points)
    : cloud_processed_(new PointCloudRGB),
      cloud_original_(new PointCloudRGB),
      view_points_(view_points) {
  sample_indices_.resize(0);
  samples_.resize(3, 0);
  normals_.resize(3, 0);

  // load and combine the two point clouds
  std::cout << "Loading point clouds ...\n";
  PointCloudRGB::Ptr cloud_left(new PointCloudRGB),
      cloud_right(new PointCloudRGB);
  cloud_left = loadPointCloudFromFile(filename_left);
  cloud_right = loadPointCloudFromFile(filename_right);

  std::cout << "Concatenating point clouds ...\n";
  *cloud_processed_ = *cloud_left + *cloud_right;
  cloud_original_ = cloud_processed_;

  std::cout << "Loaded left point cloud with " << cloud_left->size()
            << " points \n";
  std::cout << "Loaded right point cloud with " << cloud_right->size()
            << " points \n";

  // set the camera source matrix: (i,j) = 1 if point j is seen by camera i
  camera_source_ = Eigen::MatrixXi::Zero(2, cloud_processed_->size());
  camera_source_.block(0, 0, 1, cloud_left->size()) =
      Eigen::MatrixXi::Ones(1, cloud_left->size());
  camera_source_.block(1, cloud_left->size(), 1, cloud_right->size()) =
      Eigen::MatrixXi::Ones(1, cloud_right->size());
}

void Cloud::removeNans() {
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  pcl::removeNaNFromPointCloud(*cloud_processed_, inliers->indices);
  if (inliers->indices.size() < cloud_processed_->size()) {
    pcl::ExtractIndices<pcl::PointXYZRGBA> eifilter(true);
    eifilter.setInputCloud(cloud_processed_);
    eifilter.setIndices(inliers);
    eifilter.filter(*cloud_processed_);
    printf("Cloud after removing NANs: %zu\n", cloud_processed_->size());
  }
}

void Cloud::removeStatisticalOutliers() {
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;
  sor.setInputCloud(cloud_processed_);
  sor.setMeanK(50);
  sor.setStddevMulThresh(1.0);
  sor.filter(*cloud_processed_);
  printf("Cloud after removing statistical outliers: %zu\n",
         cloud_processed_->size());
}

void Cloud::refineNormals(int k) {
  std::vector<std::vector<int>> k_indices;
  std::vector<std::vector<float>> k_sqr_distances;
  pcl::search::KdTree<pcl::PointXYZRGBA> search;
  search.setInputCloud(cloud_processed_);
  search.nearestKSearch(*cloud_processed_, std::vector<int>(), k, k_indices,
                        k_sqr_distances);

  pcl::PointCloud<pcl::Normal> pcl_normals;
  pcl_normals.resize(normals_.cols());
  for (int i = 0; i < normals_.cols(); i++) {
    pcl_normals.at(i).getNormalVector3fMap() = normals_.col(i).cast<float>();
  }

  pcl::PointCloud<pcl::Normal> normals_refined;
  pcl::NormalRefinement<pcl::Normal> nr(k_indices, k_sqr_distances);
  nr.setInputCloud(pcl_normals.makeShared());
  nr.filter(normals_refined);

  Eigen::MatrixXf diff =
      normals_refined.getMatrixXfMap() - pcl_normals.getMatrixXfMap();
  printf("Refining surface normals ...\n");
  printf(" mean: %.3f, max: %.3f, sum: %.3f\n", diff.mean(), diff.maxCoeff(),
         diff.sum());

  normals_ = normals_refined.getMatrixXfMap()
                 .block(0, 0, 3, pcl_normals.size())
                 .cast<double>();
}

void Cloud::filterWorkspace(const std::vector<double> &workspace) {
  // Filter indices into the point cloud.
  if (sample_indices_.size() > 0) {
    std::vector<int> indices_to_keep;

    for (int i = 0; i < sample_indices_.size(); i++) {
      const pcl::PointXYZRGBA &p = cloud_processed_->points[sample_indices_[i]];
      if (p.x > workspace[0] && p.x < workspace[1] && p.y > workspace[2] &&
          p.y < workspace[3] && p.z > workspace[4] && p.z < workspace[5]) {
        indices_to_keep.push_back(i);
      }
    }

    sample_indices_ = indices_to_keep;
    std::cout << sample_indices_.size()
              << " sample indices left after workspace filtering \n";
  }

  // Filter (x,y,z)-samples.
  if (samples_.cols() > 0) {
    std::vector<int> indices_to_keep;

    for (int i = 0; i < samples_.cols(); i++) {
      if (samples_(0, i) > workspace[0] && samples_(0, i) < workspace[1] &&
          samples_(1, i) > workspace[2] && samples_(1, i) < workspace[3] &&
          samples_(2, i) > workspace[4] && samples_(2, i) < workspace[5]) {
        indices_to_keep.push_back(i);
      }
    }

    samples_ = EigenUtils::sliceMatrix(samples_, indices_to_keep);
    std::cout << samples_.cols()
              << " samples left after workspace filtering \n";
  }

  // Filter the point cloud.
  std::vector<int> indices;
  for (int i = 0; i < cloud_processed_->size(); i++) {
    const pcl::PointXYZRGBA &p = cloud_processed_->points[i];
    if (p.x > workspace[0] && p.x < workspace[1] && p.y > workspace[2] &&
        p.y < workspace[3] && p.z > workspace[4] && p.z < workspace[5]) {
      indices.push_back(i);
    }
  }

  Eigen::MatrixXi camera_source(camera_source_.rows(), indices.size());
  PointCloudRGB::Ptr cloud(new PointCloudRGB);
  cloud->points.resize(indices.size());
  for (int i = 0; i < indices.size(); i++) {
    camera_source.col(i) = camera_source_.col(indices[i]);
    cloud->points[i] = cloud_processed_->points[indices[i]];
  }
  if (normals_.cols() > 0) {
    Eigen::Matrix3Xd normals(3, indices.size());
    for (int i = 0; i < indices.size(); i++) {
      normals.col(i) = normals_.col(indices[i]);
    }
    normals_ = normals;
  }
  cloud_processed_ = cloud;
  camera_source_ = camera_source;
}

void Cloud::filterSamples(const std::vector<double> &workspace) {
  std::vector<int> indices;
  for (int i = 0; i < samples_.size(); i++) {
    if (samples_(0, i) > workspace[0] && samples_(0, i) < workspace[1] &&
        samples_(1, i) > workspace[2] && samples_(1, i) < workspace[3] &&
        samples_(2, i) > workspace[4] && samples_(2, i) < workspace[5]) {
      indices.push_back(i);
    }
  }

  Eigen::Matrix3Xd filtered_samples(3, indices.size());
  for (int i = 0; i < indices.size(); i++) {
    filtered_samples.col(i) = samples_.col(i);
  }
  samples_ = filtered_samples;
}

void Cloud::voxelizeCloud(float cell_size) {
  // Find the cell that each point falls into.
  pcl::PointXYZRGBA min_pt_pcl;
  pcl::PointXYZRGBA max_pt_pcl;
  pcl::getMinMax3D(*cloud_processed_, min_pt_pcl, max_pt_pcl);
  const Eigen::Vector3f min_pt = min_pt_pcl.getVector3fMap();
  std::set<Eigen::Vector4i, Cloud::UniqueVector4First3Comparator> bins;
  Eigen::Matrix3Xd avg_normals =
      Eigen::Matrix3Xd::Zero(3, cloud_processed_->size());
  Eigen::VectorXi counts = Eigen::VectorXi::Zero(cloud_processed_->size());

  for (int i = 0; i < cloud_processed_->size(); i++) {
    const Eigen::Vector3f pt = cloud_processed_->at(i).getVector3fMap();
    Eigen::Vector4i v4;
    v4.head(3) = EigenUtils::floorVector((pt - min_pt) / cell_size);
    v4(3) = i;
    std::pair<std::set<Eigen::Vector4i,
                       Cloud::UniqueVector4First3Comparator>::iterator,
              bool>
        res = bins.insert(v4);

    if (normals_.cols() > 0) {
      const int &idx = (*res.first)(3);
      avg_normals.col(idx) += normals_.col(i);
      counts(idx)++;
    }
  }

  // Calculate the point value and the average surface normal for each cell, and
  // set the camera source for each point.
  Eigen::Matrix3Xf voxels(3, bins.size());
  Eigen::Matrix3Xd normals(3, bins.size());
  Eigen::MatrixXi camera_source(camera_source_.rows(), bins.size());
  int i = 0;
  std::set<Eigen::Vector4i, Cloud::UniqueVector4First3Comparator>::iterator it;

  for (it = bins.begin(); it != bins.end(); it++) {
    voxels.col(i) = min_pt + cell_size * (*it).head(3).cast<float>();
    const int &idx = (*it)(3);

    for (int j = 0; j < camera_source_.rows(); j++) {
      camera_source(j, i) = (camera_source_(j, idx) == 1) ? 1 : 0;
    }
    if (normals_.cols() > 0) {
      normals.col(i) = avg_normals.col(idx) / (double)counts(idx);
    }
    i++;
  }

  // Copy the voxels into the point cloud.
  cloud_processed_->points.resize(voxels.cols());
  for (int i = 0; i < voxels.cols(); i++) {
    cloud_processed_->points[i].getVector3fMap() = voxels.col(i);
  }

  camera_source_ = camera_source;

  if (normals_.cols() > 0) {
    normals_ = normals;
  }

  printf("Voxelized cloud: %zu\n", cloud_processed_->size());
}

void Cloud::subsample(int num_samples) {
  if (num_samples == 0) {
    return;
  }

  if (samples_.cols() > 0) {
    subsampleSamples(num_samples);
  } else if (sample_indices_.size() > 0) {
    subsampleSampleIndices(num_samples);
  } else {
    subsampleUniformly(num_samples);
  }
}

void Cloud::subsampleUniformly(int num_samples) {
  sample_indices_.resize(num_samples);
  pcl::RandomSample<pcl::PointXYZRGBA> random_sample;
  random_sample.setInputCloud(cloud_processed_);
  random_sample.setSample(num_samples);
  random_sample.filter(sample_indices_);
}

void Cloud::subsampleSamples(int num_samples) {
  if (num_samples == 0 || num_samples >= samples_.cols()) {
    return;
  } else {
    std::cout << "Using " << num_samples << " out of " << samples_.cols()
              << " available samples.\n";
    std::vector<int> seq(samples_.cols());
    for (int i = 0; i < seq.size(); i++) {
      seq[i] = i;
    }
    std::random_shuffle(seq.begin(), seq.end());

    Eigen::Matrix3Xd subsamples(3, num_samples);
    for (int i = 0; i < num_samples; i++) {
      subsamples.col(i) = samples_.col(seq[i]);
    }
    samples_ = subsamples;

    std::cout << "Subsampled " << samples_.cols()
              << " samples at random uniformly.\n";
  }
}

void Cloud::subsampleSampleIndices(int num_samples) {
  if (sample_indices_.size() == 0 || num_samples >= sample_indices_.size()) {
    return;
  }

  std::vector<int> indices(num_samples);
  for (int i = 0; i < num_samples; i++) {
    indices[i] = sample_indices_[rand() % sample_indices_.size()];
  }
  sample_indices_ = indices;
}

void Cloud::sampleAbovePlane() {
  double t0 = omp_get_wtime();
  printf("Sampling above plane ...\n");
  std::vector<int> indices(0);
  pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  seg.setInputCloud(cloud_processed_);
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);
  seg.segment(*inliers, *coefficients);
  if (inliers->indices.size() > 0) {
    pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
    extract.setInputCloud(cloud_processed_);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(indices);
  }
  if (indices.size() > 0) {
    sample_indices_ = indices;
    printf(" Plane fit succeeded. %zu samples above plane.\n",
           sample_indices_.size());
  } else {
    printf(" Plane fit failed. Using entire point cloud ...\n");
  }
  std::cout << " runtime (plane fit): " << omp_get_wtime() - t0 << "\n";
}

void Cloud::writeNormalsToFile(const std::string &filename,
                               const Eigen::Matrix3Xd &normals) {
  std::ofstream myfile;
  myfile.open(filename.c_str());

  for (int i = 0; i < normals.cols(); i++) {
    myfile << std::to_string(normals(0, i)) << ","
           << std::to_string(normals(1, i)) << ","
           << std::to_string(normals(2, i)) << "\n";
  }

  myfile.close();
}

void Cloud::calculateNormals(int num_threads, double radius) {
  double t_gpu = omp_get_wtime();
  printf("Calculating surface normals ...\n");
  std::string mode;

#if defined(USE_PCL_GPU)
  calculateNormalsGPU();
  mode = "gpu";
#else
  if (cloud_processed_->isOrganized()) {
    calculateNormalsOrganized();
    mode = "integral images";
  } else {
    printf("num_threads: %d\n", num_threads);
    calculateNormalsOMP(num_threads, radius);
    mode = "OpenMP";
  }
#endif

  t_gpu = omp_get_wtime() - t_gpu;
  printf("Calculated %zu surface normals in %3.4fs (mode: %s).\n",
         normals_.cols(), t_gpu, mode.c_str());
  printf(
      "Reversing direction of normals that do not point to at least one camera "
      "...\n");
  reverseNormals();
}

void Cloud::calculateNormalsOrganized() {
  if (!cloud_processed_->isOrganized()) {
    std::cout << "Error: point cloud is not organized!\n";
    return;
  }

  std::cout << "Using integral images for surface normals estimation ...\n";
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
      new pcl::PointCloud<pcl::Normal>);
  pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
  ne.setInputCloud(cloud_processed_);
  ne.setViewPoint(view_points_(0, 0), view_points_(1, 0), view_points_(2, 0));
  ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
  ne.setNormalSmoothingSize(20.0f);
  ne.compute(*cloud_normals);
  normals_ = cloud_normals->getMatrixXfMap().cast<double>();
}

void Cloud::calculateNormalsOMP(int num_threads, double radius) {
  std::vector<std::vector<int>> indices = convertCameraSourceMatrixToLists();

  // Calculate surface normals for each view point.
  std::vector<PointCloudNormal::Ptr> normals_list(view_points_.cols());
  pcl::NormalEstimationOMP<pcl::PointXYZRGBA, pcl::Normal> estimator(
      num_threads);
  pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree_ptr(
      new pcl::search::KdTree<pcl::PointXYZRGBA>);
  estimator.setInputCloud(cloud_processed_);
  estimator.setSearchMethod(tree_ptr);
  estimator.setRadiusSearch(radius);
  pcl::IndicesPtr indices_ptr(new std::vector<int>);

  for (int i = 0; i < view_points_.cols(); i++) {
    PointCloudNormal::Ptr normals_cloud(new PointCloudNormal);
    indices_ptr->assign(indices[i].begin(), indices[i].end());
    estimator.setIndices(indices_ptr);
    estimator.setViewPoint(view_points_(0, i), view_points_(1, i),
                           view_points_(2, i));
    double t0 = omp_get_wtime();
    estimator.compute(*normals_cloud);
    printf(" runtime(computeNormals): %3.4f\n", omp_get_wtime() - t0);
    normals_list[i] = normals_cloud;
    printf("camera: %d, #indices: %d, #normals: %d \n", i,
           (int)indices[i].size(), (int)normals_list[i]->size());
  }

  // Assign the surface normals to the points.
  normals_.resize(3, camera_source_.cols());

  for (int i = 0; i < normals_list.size(); i++) {
    for (int j = 0; j < normals_list[i]->size(); j++) {
      const pcl::Normal &normal = normals_list[i]->at(j);
      normals_.col(indices[i][j]) << normal.normal_x, normal.normal_y,
          normal.normal_z;
    }
  }
}

#if defined(USE_PCL_GPU)
void Cloud::calculateNormalsGPU() {
  std::vector<std::vector<int>> indices = convertCameraSourceMatrixToLists();

  PointCloudXYZ::Ptr cloud_xyz(new PointCloudXYZ);
  pcl::copyPointCloud(*cloud_processed_, *cloud_xyz);
  pcl::gpu::Feature::PointCloud cloud_device;
  cloud_device.upload(cloud_xyz->points);
  pcl::gpu::Feature::Normals normals_device;
  pcl::gpu::NormalEstimation ne;
  ne.setInputCloud(cloud_device);
  // ne.setRadiusSearch(0.03, 1000);
  ne.setRadiusSearch(0.03, 2000);
  // ne.setRadiusSearch(0.03, 4000);
  // ne.setRadiusSearch(0.03, 8000);
  pcl::gpu::Feature::Indices indices_device;
  std::vector<pcl::PointXYZ> downloaded;
  normals_.resize(3, camera_source_.cols());

  // Calculate surface normals for each view point.
  for (int i = 0; i < view_points_.cols(); i++) {
    const Eigen::Vector3d &view_point = view_points_.col(i);
    indices_device.upload(indices[i]);
    ne.setViewPoint(view_point(0), view_point(1), view_point(2));
    ne.setIndices(indices_device);
    ne.compute(normals_device);
    normals_device.download(downloaded);

    for (int j = 0; j < indices[i].size(); j++) {
      normals_.col(indices[i][j]) =
          downloaded[i].getVector3fMap().cast<double>();
    }
  }
}
#endif

void Cloud::reverseNormals() {
  double t1 = omp_get_wtime();
  int c = 0;

  for (int i = 0; i < normals_.cols(); i++) {
    bool needs_reverse = true;

    for (int j = 0; j < view_points_.cols(); j++) {
      if (camera_source_(j, i) == 1)  // point is seen by this camera
      {
        Eigen::Vector3d cam_to_point =
            cloud_processed_->at(i).getVector3fMap().cast<double>() -
            view_points_.col(j);

        if (normals_.col(i).dot(cam_to_point) <
            0)  // normal points toward camera
        {
          needs_reverse = false;
          break;
        }
      }
    }

    if (needs_reverse) {
      normals_.col(i) *= -1.0;
      c++;
    }
  }

  std::cout << " reversed " << c << " normals\n";
  std::cout << " runtime (reverse normals): " << omp_get_wtime() - t1 << "\n";
}

std::vector<std::vector<int>> Cloud::convertCameraSourceMatrixToLists() {
  std::vector<std::vector<int>> indices(view_points_.cols());

  for (int i = 0; i < camera_source_.cols(); i++) {
    for (int j = 0; j < view_points_.cols(); j++) {
      if (camera_source_(j, i) == 1)  // point is seen by this camera
      {
        indices[j].push_back(i);
        break;  // TODO: multiple cameras
      }
    }
  }

  return indices;
}

void Cloud::setNormalsFromFile(const std::string &filename) {
  std::ifstream in;
  in.open(filename.c_str());
  std::string line;
  normals_.resize(3, cloud_original_->size());
  int i = 0;

  while (std::getline(in, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    int j = 0;

    while (std::getline(lineStream, cell, ',')) {
      normals_(i, j) = std::stod(cell);
      j++;
    }

    i++;
  }
}

PointCloudRGB::Ptr Cloud::loadPointCloudFromFile(
    const std::string &filename) const {
  PointCloudRGB::Ptr cloud(new PointCloudRGB);
  std::string extension = filename.substr(filename.size() - 3);
  printf("extension: %s\n", extension.c_str());

  if (extension == "pcd" &&
      pcl::io::loadPCDFile<pcl::PointXYZRGBA>(filename, *cloud) == -1) {
    printf("Couldn't read PCD file: %s\n", filename.c_str());
    cloud->points.resize(0);
  } else if (extension == "ply" &&
             pcl::io::loadPLYFile<pcl::PointXYZRGBA>(filename, *cloud) == -1) {
    printf("Couldn't read PLY file: %s\n", filename.c_str());
    cloud->points.resize(0);
  }

  return cloud;
}

void Cloud::setSamples(const Eigen::Matrix3Xd &samples) { samples_ = samples; }

}  // namespace util
}  // namespace gpd
