#include <gpd/learning.h>


void Learning::extractGraspsAndImages(const std::vector<GraspSet>& hand_set_list, const std::vector<cv::Mat>& images,
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


std::vector<cv::Mat> Learning::createImages(const CloudCamera& cloud_cam, const std::vector<GraspSet>& hand_set_list)
  const
{
  double t0_total = omp_get_wtime();

  PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
  *cloud = *cloud_cam.getCloudProcessed();
  Eigen::Matrix3Xd points = cloud->getMatrixXfMap().cast<double>();
  PointList point_list(points, cloud_cam.getNormals(), cloud_cam.getCameraSource(), cloud_cam.getViewPoints());

  if (remove_plane_)
  {
	  // Segment the plane to speed up shadow computation.
    pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    seg.setInputCloud(cloud_cam.getCloudProcessed());
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
//    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.01);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size () > 0)
    {
      pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
      extract.setInputCloud(cloud_cam.getCloudProcessed());
      extract.setIndices(inliers);
      extract.setNegative(true);
      std::vector<int> indices;
      extract.filter(indices);
//      pcl::visualization::CloudViewer viewer("Filtered Cloud");
//      viewer.showCloud(cloud);
//      while (!viewer.wasStopped()) { }
      if (indices.size() > 0)
      {
        extract.filter(*cloud);
        point_list = point_list.slice(indices);
        printf("Removed plane from point cloud. Remaining points in cloud: %d.\n", (int) cloud->size());
        std::cout << point_list.getPoints().col(0).transpose() << "\n" << cloud->at(0).getVector3fMap().transpose() << "\n";
      }
      else
      {
        printf("No points remaining after plane is removed! Using entire point cloud ...\n");
      }
    }
  }

  // Prepare kd-tree for neighborhood searches in the point cloud.
  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud);
  std::vector<int> nn_indices;
  std::vector<float> nn_dists;

  // Set the radius for the neighborhood search to the largest image dimension.
  Eigen::Vector3d image_dims;
  image_dims << image_params_.depth_, image_params_.height_ / 2.0, image_params_.outer_diameter_;
  double radius = image_dims.maxCoeff();

  // 1. Find points within image dimensions.
  bool is_valid[hand_set_list.size()];
  std::vector<PointList> nn_points_list;
  nn_points_list.resize(hand_set_list.size());

#ifdef _OPENMP // parallelization using OpenMP
#pragma omp parallel for private(nn_indices, nn_dists) num_threads(num_threads_)
#endif
  for (int i = 0; i < hand_set_list.size(); i++)
  {
    pcl::PointXYZRGBA sample_pcl;
    sample_pcl.getVector3fMap() = hand_set_list[i].getSample().cast<float>();

    if (kdtree.radiusSearch(sample_pcl, radius, nn_indices, nn_dists) > 0)
    {
      nn_points_list[i] = point_list.slice(nn_indices);
      is_valid[i] = true;
    }
    else
    {
      is_valid[i] = false;
    }
  }

  std::cout << "time for computing " << nn_points_list.size() << " point neighborhoods with " << num_threads_ << " threads: " << omp_get_wtime() - t0_total << "s\n";

  if (image_params_.num_channels_ == 1 || image_params_.num_channels_ == 3) // 3 channels image (only surface normals)
  {
    return createImages1or3Channels(hand_set_list, nn_points_list, is_valid, image_dims);
  }
  else if (image_params_.num_channels_ == 15) // 15 channels image
  {
    return createImages15Channels(hand_set_list, nn_points_list, is_valid, image_dims);
  }

  std::vector<cv::Mat> empty;
  empty.resize(0);
  return empty;
}


std::vector<cv::Mat> Learning::createImages1or3Channels(const std::vector<GraspSet>& hand_set_list,
  const std::vector<PointList>& nn_points_list, const bool* is_valid, const Eigen::Vector3d& image_dims) const
{
  double t0_images = omp_get_wtime();
  std::vector<cv::Mat> images(hand_set_list.size() * num_orientations_, cv::Mat(60, 60, CV_8UC(image_params_.num_channels_), cv::Scalar(0.0)));

#ifdef _OPENMP // parallelization using OpenMP
#pragma omp parallel for num_threads(num_threads_)
#endif
  for (int i = 0; i < hand_set_list.size(); i++)
  {
    if (is_valid[i])
    {
      const std::vector<Grasp>& hands = hand_set_list[i].getHypotheses();

      for (int j = 0; j < hands.size(); j++)
      {
        if (hand_set_list[i].getIsValid()(j))
        {
          const int idx = i * num_orientations_ + j;
          images[idx] = createImage1or3Channels(nn_points_list[i], hands[j]);
        }
      }
    }
  }

  return images;
}


cv::Mat Learning::createImage1or3Channels(const PointList& point_list, const Grasp& hand) const
{
  // 1. Transform points and normals in neighborhood into the unit image.
  Matrix3XdPair points_normals = transformToUnitImage(point_list, hand);

  // 2. Calculate grasp image.
  Eigen::VectorXi cell_indices = GraspImage::findCellIndices(points_normals.first);
  GraspImage::setImageSize(image_params_.size_);
  cv::Mat image;

  if (image_params_.num_channels_ == 1)
  {
    image = GraspImage::createBinaryImage(cell_indices);
  }
  else if (image_params_.num_channels_ == 3)
  {
    image = GraspImage::createNormalsImage(points_normals.second, cell_indices);
  }

  if (is_plotting_)
  {
    std::string title = "Grasp Image (" + boost::lexical_cast<std::string>(image_params_.num_channels_);
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, image);
    cv::waitKey(0);
  }

  return image;
}


std::vector<cv::Mat> Learning::createImages15Channels(const std::vector<GraspSet>& hand_set_list,
  const std::vector<PointList>& nn_points_list, const bool* is_valid, const Eigen::Vector3d& image_dims) const
{
  // 1. Calculate shadow points for each point neighborhood.
  double t0_shadows = omp_get_wtime();

  // Calculate shadow length (= max length of shadow required to fill image window).
  double shadow_length = image_dims.maxCoeff();

  std::vector<cv::Mat> images(hand_set_list.size() * num_orientations_, cv::Mat(60, 60, CV_8UC(15), cv::Scalar(0.0)));

#ifdef _OPENMP // parallelization using OpenMP
#pragma omp parallel for num_threads(num_threads_)
#endif
  for (int i = 0; i < nn_points_list.size(); i++)
  {
    if (is_valid[i])
    {
      Eigen::Matrix3Xd shadow = hand_set_list[i].calculateShadow4(nn_points_list[i], shadow_length);

      const std::vector<Grasp>& hands = hand_set_list[i].getHypotheses();

      for (int j = 0; j < hands.size(); j++)
      {
        if (hand_set_list[i].getIsValid()(j))
        {
          const int idx = i * num_orientations_ + j;
          images[idx] = createImage15Channels(nn_points_list[i], shadow, hands[j]);
        }
      }
    }
  }

  return images;
}


cv::Mat Learning::createImage15Channels(const PointList& point_list, const Eigen::Matrix3Xd& shadow, const Grasp& hand)
  const
{
  // 1. Transform points and normals in neighborhood into the unit image.
  Matrix3XdPair points_normals = transformToUnitImage(point_list, hand);

  // 2. Transform occluded points into hand frame.
  Eigen::Matrix3Xd shadow_frame = shadow - hand.getSample().replicate(1, shadow.cols());
  shadow_frame = hand.getFrame().transpose() * shadow_frame;
  std::vector<int> indices = findPointsInUnitImage(hand, shadow_frame);
  Eigen::Matrix3Xd cropped_shadow_points = transformPointsToUnitImage(hand, shadow_frame, indices);

  // 3. Create grasp image.
  GraspImage15Channels grasp_image(image_params_.size_, is_plotting_, &points_normals.first, &points_normals.second,
    &cropped_shadow_points);
  cv::Mat image = grasp_image.calculateImage();
  return image;
}


Matrix3XdPair Learning::transformToUnitImage(const PointList& point_list, const Grasp& hand) const
{
  // 1. Transform points and normals in neighborhood into the hand frame.
  Matrix3XdPair points_normals = transformToHandFrame(point_list, hand.getSample(), hand.getFrame().transpose());

  // 2. Find points in unit image.
  std::vector<int> indices = findPointsInUnitImage(hand, points_normals.first);
  points_normals.first = transformPointsToUnitImage(hand, points_normals.first, indices);
  points_normals.second = EigenUtils::sliceMatrix(points_normals.second, indices);

  return points_normals;
}


std::vector<int> Learning::findPointsInUnitImage(const Grasp& hand, const Eigen::Matrix3Xd& points) const
{
  std::vector<int> indices;
  const double half_outer_diameter = image_params_.outer_diameter_ / 2.0;

  for (int i = 0; i < points.cols(); i++)
  {
    if ( (points(0,i) > hand.getBottom()) && (points(0,i) < hand.getBottom() + image_params_.depth_)
      && (points(1,i) > hand.getCenter() - half_outer_diameter)
      && (points(1,i) < hand.getCenter() + half_outer_diameter)
      && (points(2,i) > -1.0 * image_params_.height_) && (points(2,i) < image_params_.height_))
    {
      indices.push_back(i);
    }
  }

  return indices;
}


Eigen::Matrix3Xd Learning::transformPointsToUnitImage(const Grasp& hand, const Eigen::Matrix3Xd& points,
  const std::vector<int>& indices) const
{
  Eigen::Matrix3Xd points_out(3, indices.size());
  const double half_outer_diameter = image_params_.outer_diameter_ / 2.0;
  const double double_height = 2.0 * image_params_.height_;

  for (int i = 0; i < indices.size(); i++)
  {
    points_out(0,i) = (points(0,indices[i]) - hand.getBottom()) / image_params_.depth_;
    points_out(1,i) = (points(1,indices[i]) - (hand.getCenter() - half_outer_diameter)) / image_params_.outer_diameter_;
    points_out(2,i) = (points(2,indices[i]) + image_params_.height_) / double_height;
  }

  return points_out;
}


Matrix3XdPair Learning::transformToHandFrame(const PointList& point_list, const Eigen::Vector3d& centroid,
  const Eigen::Matrix3d& rotation) const
{
  Matrix3XdPair pair;
  pair.first = rotation * (point_list.getPoints() - centroid.replicate(1, point_list.getPoints().cols()));
  pair.second = rotation * point_list.getNormals();
  return pair;
}
