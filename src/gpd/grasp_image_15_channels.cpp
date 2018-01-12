#include <gpd/grasp_image_15_channels.h>


const int GraspImage15Channels::NUM_CHANNELS = 15;


GraspImage15Channels::GraspImage15Channels(int image_size, bool is_plotting, Eigen::Matrix3Xd* points,
  Eigen::Matrix3Xd* normals, Eigen::Matrix3Xd* shadow)
  : GraspImage(image_size, NUM_CHANNELS, is_plotting), points_(points), normals_(normals), shadow_(shadow)
{

}


GraspImage15Channels::~GraspImage15Channels()
{

}


cv::Mat GraspImage15Channels::calculateImage()
{
  // 1. Create the images for the first projection.
  Eigen::Vector3i order;
  order << 0, 1, 2;
  double t0 = omp_get_wtime();
  Projection projection1 = calculateProjection(*points_, *normals_, *shadow_);
//  std::cout << "   time for view 1: " << omp_get_wtime() - t0 << "\n";

  Eigen::Matrix3Xd points = *points_;
  Eigen::Matrix3Xd shadow = *shadow_;

  // 2. Create the images for the second view.
  order << 2, 1, 0;
  t0 = omp_get_wtime();
  points.row(0).swap(points.row(2));
  shadow.row(0).swap(shadow.row(2));
  Projection projection2 = calculateProjection(points, *normals_, shadow);
//  std::cout << "   time for view 2: " << omp_get_wtime() - t0 << "\n";

  // 3. Create the images for the third view.
  order << 2, 0, 1;
  t0 = omp_get_wtime();
  points.row(1).swap(points.row(2));
  shadow.row(1).swap(shadow.row(2));
  Projection projection3 = calculateProjection(points, *normals_, shadow);
//  std::cout << "   time for view 3: " << omp_get_wtime() - t0 << "\n";

  // 4. Concatenate the images from the three projections into a single image.
  t0 = omp_get_wtime();
  cv::Mat img = concatenateProjections(projection1, projection2, projection3);
//  std::cout << "   time for concatenating views: " << omp_get_wtime() - t0 << "\n";

  // Visualize the final image.
  if (is_plotting_)
  {
    std::vector<Projection> projections;
    projections.push_back(projection1);
    projections.push_back(projection2);
    projections.push_back(projection3);
    showImage(projections);
  }

  return img;
}


Projection GraspImage15Channels::calculateProjection(const Eigen::Matrix3Xd& points, const Eigen::Matrix3Xd& normals,
  const Eigen::Matrix3Xd& shadow)
{
  double t0 = omp_get_wtime();

  Projection projection;

  Eigen::VectorXi cell_indices = findCellIndices(points);
  //  std::cout << " time for finding cell indices: " << omp_get_wtime() - t0 << "s\n";

  t0 = omp_get_wtime();
  projection.normals_image_ = createNormalsImage(normals, cell_indices);
  //  std::cout << " time for normals image: " << omp_get_wtime() - t0 << "s\n";

  t0 = omp_get_wtime();
  projection.depth_image_ = createDepthImage(points, cell_indices);
  //  std::cout << " time for depth image: " << omp_get_wtime() - t0 << "s\n";

  t0 = omp_get_wtime();
  cell_indices = findCellIndices(shadow);
  projection.shadow_image_ = createShadowImage(shadow, cell_indices);
  //  std::cout << " time for shadow image: " << omp_get_wtime() - t0 << "s\n";

  return projection;
}


cv::Mat GraspImage15Channels::concatenateProjections(const Projection& projection1, const Projection& projection2,
  const Projection& projection3) const
{
  cv::Mat image(GraspImage::image_size_, GraspImage::image_size_, CV_8UC(num_channels_));

  // Not sure if this is optimal but it avoids loops for reducing code length.
  std::vector<cv::Mat> projection1_list, projection2_list, projection3_list;
  cv::split(projection1.normals_image_, projection1_list);
  cv::split(projection2.normals_image_, projection2_list);
  cv::split(projection3.normals_image_, projection3_list);
  projection1_list.push_back(projection1.depth_image_);
  projection1_list.push_back(projection1.shadow_image_);
  projection1_list.push_back(projection2_list[0]);
  projection1_list.push_back(projection2_list[1]);
  projection1_list.push_back(projection2_list[2]);
  projection1_list.push_back(projection2.depth_image_);
  projection1_list.push_back(projection2.shadow_image_);
  projection1_list.push_back(projection3_list[0]);
  projection1_list.push_back(projection3_list[1]);
  projection1_list.push_back(projection3_list[2]);
  projection1_list.push_back(projection3.depth_image_);
  projection1_list.push_back(projection3.shadow_image_);
  cv::merge(projection1_list, image);

  return image;
}


void GraspImage15Channels::showImage(const std::vector<Projection>& projections) const
{
  int border = 5;
  int n = 3;
  int total_size = n * (GraspImage::image_size_ + border) + border;
  cv::Mat image_out(total_size, total_size, CV_8UC3, cv::Scalar(0.5));

  for (int i = 0; i < n; i++)
  {
    // OpenCV requires images to be in BGR or grayscale to be displayed.
    cv::Mat normals_rgb, depth_rgb, shadow_rgb;
    cvtColor(projections[i].normals_image_, normals_rgb, CV_RGB2BGR);
    cvtColor(projections[i].depth_image_, depth_rgb, CV_GRAY2RGB);
    cvtColor(projections[i].shadow_image_, shadow_rgb, CV_GRAY2RGB);
    normals_rgb.copyTo(image_out(cv::Rect(border, border + i * (border + image_size_), image_size_, image_size_)));
    depth_rgb.copyTo(image_out(cv::Rect(2 * border + image_size_, border + i * (border + image_size_), image_size_, image_size_)));
    shadow_rgb.copyTo(image_out(cv::Rect(3 * border + 2 * image_size_, border + i * (border + image_size_), image_size_, image_size_)));
  }

  cv::namedWindow("Grasp Image (15 channels)", cv::WINDOW_NORMAL);
  cv::imshow("Grasp Image (15 channels)", image_out);
  cv::waitKey(0);
}
