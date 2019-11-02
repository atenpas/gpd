#include <gpd/descriptor/image_generator.h>

namespace gpd {
namespace descriptor {

ImageGenerator::ImageGenerator(const descriptor::ImageGeometry &image_geometry,
                               int num_threads, int num_orientations,
                               bool is_plotting, bool remove_plane)
    : image_params_(image_geometry),
      num_threads_(num_threads),
      num_orientations_(num_orientations),
      remove_plane_(remove_plane) {
  image_strategy_ = descriptor::ImageStrategy::makeImageStrategy(
      image_geometry, num_threads_, num_orientations_, is_plotting);
}

void ImageGenerator::createImages(
    const util::Cloud &cloud_cam,
    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    std::vector<std::unique_ptr<cv::Mat>> &images_out,
    std::vector<std::unique_ptr<candidate::Hand>> &hands_out) const {
  double t0 = omp_get_wtime();

  Eigen::Matrix3Xd points =
      cloud_cam.getCloudProcessed()->getMatrixXfMap().cast<double>().block(
          0, 0, 3, cloud_cam.getCloudProcessed()->points.size());
  util::PointList point_list(points, cloud_cam.getNormals(),
                             cloud_cam.getCameraSource(),
                             cloud_cam.getViewPoints());

  // Segment the support/table plane to speed up shadow computation.
  if (remove_plane_) {
    removePlane(cloud_cam, point_list);
  }

  // Prepare kd-tree for neighborhood searches in the point cloud.
  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud_cam.getCloudProcessed());
  std::vector<int> nn_indices;
  std::vector<float> nn_dists;

  // Set the radius for the neighborhood search to the largest image dimension.
  Eigen::Vector3d image_dims;
  image_dims << image_params_.depth_, image_params_.height_ / 2.0,
      image_params_.outer_diameter_;
  double radius = image_dims.maxCoeff();

  // 1. Find points within image dimensions.
  std::vector<util::PointList> nn_points_list;
  nn_points_list.resize(hand_set_list.size());

  double t_slice = omp_get_wtime();

#ifdef _OPENMP  // parallelization using OpenMP
#pragma omp parallel for private(nn_indices, nn_dists) num_threads(num_threads_)
#endif
  for (int i = 0; i < hand_set_list.size(); i++) {
    pcl::PointXYZRGBA sample_pcl;
    sample_pcl.getVector3fMap() = hand_set_list[i]->getSample().cast<float>();

    if (kdtree.radiusSearch(sample_pcl, radius, nn_indices, nn_dists) > 0) {
      nn_points_list[i] = point_list.slice(nn_indices);
    }
  }
  printf("neighborhoods search time: %3.4f\n", omp_get_wtime() - t_slice);

  createImageList(hand_set_list, nn_points_list, images_out, hands_out);
  printf("Created %zu images in %3.4fs\n", images_out.size(),
         omp_get_wtime() - t0);
}

void ImageGenerator::createImageList(
    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    const std::vector<util::PointList> &nn_points_list,
    std::vector<std::unique_ptr<cv::Mat>> &images_out,
    std::vector<std::unique_ptr<candidate::Hand>> &hands_out) const {
  double t0_images = omp_get_wtime();

  int m = hand_set_list[0]->getHands().size();
  int n = hand_set_list.size() * m;
  std::vector<std::vector<std::unique_ptr<cv::Mat>>> images_list(n);

#ifdef _OPENMP  // parallelization using OpenMP
#pragma omp parallel for num_threads(num_threads_)
#endif
  for (int i = 0; i < hand_set_list.size(); i++) {
    images_list[i] =
        image_strategy_->createImages(*hand_set_list[i], nn_points_list[i]);
  }

  for (int i = 0; i < hand_set_list.size(); i++) {
    for (int j = 0; j < hand_set_list[i]->getHands().size(); j++) {
      if (hand_set_list[i]->getIsValid()(j)) {
        images_out.push_back(std::move(images_list[i][j]));
        hands_out.push_back(std::move(hand_set_list[i]->getHands()[j]));
      }
    }
  }
}

void ImageGenerator::removePlane(const util::Cloud &cloud_cam,
                                 util::PointList &point_list) const {
  pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  seg.setInputCloud(cloud_cam.getCloudProcessed());
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);
  seg.segment(*inliers, *coefficients);
  if (inliers->indices.size() > 0) {
    pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
    extract.setInputCloud(cloud_cam.getCloudProcessed());
    extract.setIndices(inliers);
    extract.setNegative(true);
    std::vector<int> indices;
    extract.filter(indices);
    if (indices.size() > 0) {
      PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
      extract.filter(*cloud);
      point_list = point_list.slice(indices);
      printf("Removed plane from point cloud. %zu points remaining.\n",
             cloud->size());
    } else {
      printf("Plane fit failed. Using entire point cloud ...\n");
    }
  }
}

}  // namespace descriptor
}  // namespace gpd
