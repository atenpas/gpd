#include <nodes/grasp_detection_server.h>


GraspDetectionServer::GraspDetectionServer(ros::NodeHandle& node)
{
  cloud_camera_ = NULL;

  // set camera viewpoint to default origin
  std::vector<double> camera_position;
  node.getParam("camera_position", camera_position);
  view_point_ << camera_position[0], camera_position[1], camera_position[2];

  grasp_detector_ = new GraspDetector(node);

  std::string rviz_topic;
  node.param("rviz_topic", rviz_topic, std::string(""));

  if (!rviz_topic.empty())
  {
    rviz_plotter_ = new GraspPlotter(node, grasp_detector_->getHandSearchParameters());
    use_rviz_ = true;
  }
  else
  {
    use_rviz_ = false;
  }

  // Advertise ROS topic for detected grasps.
  grasps_pub_ = node.advertise<gpd::GraspConfigList>("clustered_grasps", 10);

  node.getParam("workspace", workspace_);
}


bool GraspDetectionServer::detectGrasps(gpd::detect_grasps::Request& req, gpd::detect_grasps::Response& res)
{
  ROS_INFO("Received service request ...");

  // 1. Initialize cloud camera.
  cloud_camera_ = NULL;
  const gpd::CloudSources& cloud_sources = req.cloud_indexed.cloud_sources;

  // Set view points.
  Eigen::Matrix3Xd view_points(3, cloud_sources.view_points.size());
  for (int i = 0; i < cloud_sources.view_points.size(); i++)
  {
    view_points.col(i) << cloud_sources.view_points[i].x, cloud_sources.view_points[i].y,
      cloud_sources.view_points[i].z;
  }

  // Set point cloud.
  if (cloud_sources.cloud.fields.size() == 6 && cloud_sources.cloud.fields[3].name == "normal_x"
    && cloud_sources.cloud.fields[4].name == "normal_y" && cloud_sources.cloud.fields[5].name == "normal_z")
  {
    PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
    pcl::fromROSMsg(cloud_sources.cloud, *cloud);

    // TODO: multiple cameras can see the same point
    Eigen::MatrixXi camera_source = Eigen::MatrixXi::Zero(view_points.cols(), cloud->size());
    for (int i = 0; i < cloud_sources.camera_source.size(); i++)
    {
      camera_source(cloud_sources.camera_source[i].data, i) = 1;
    }

    cloud_camera_ = new CloudCamera(cloud, camera_source, view_points);
  }
  else
  {
    PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
    pcl::fromROSMsg(cloud_sources.cloud, *cloud);

    // TODO: multiple cameras can see the same point
    Eigen::MatrixXi camera_source = Eigen::MatrixXi::Zero(view_points.cols(), cloud->size());
    for (int i = 0; i < cloud_sources.camera_source.size(); i++)
    {
      camera_source(cloud_sources.camera_source[i].data, i) = 1;
    }

    cloud_camera_ = new CloudCamera(cloud, camera_source, view_points);
    std::cout << "view_points:\n" << view_points << "\n";
  }

  // Set the indices at which to sample grasp candidates.
  std::vector<int> indices(req.cloud_indexed.indices.size());
  for (int i=0; i < indices.size(); i++)
  {
    indices[i] = req.cloud_indexed.indices[i].data;
  }
  cloud_camera_->setSampleIndices(indices);

  frame_ = req.cloud_indexed.cloud_sources.cloud.header.frame_id;

  ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points, and "
    << req.cloud_indexed.indices.size() << " samples");

  // 2. Preprocess the point cloud.
  grasp_detector_->preprocessPointCloud(*cloud_camera_);

  // 3. Detect grasps in the point cloud.
  std::vector<Grasp> grasps = grasp_detector_->detectGrasps(*cloud_camera_);

  if (grasps.size() > 0)
  {
    // Visualize the detected grasps in rviz.
    if (use_rviz_)
    {
      rviz_plotter_->drawGrasps(grasps, frame_);
    }

    // Publish the detected grasps.
    gpd::GraspConfigList selected_grasps_msg = createGraspListMsg(grasps);
    res.grasp_configs = selected_grasps_msg;
    ROS_INFO_STREAM("Detected " << selected_grasps_msg.grasps.size() << " highest-scoring grasps.");
    return true;
  }

  ROS_WARN("No grasps detected!");
  return false;
}


gpd::GraspConfigList GraspDetectionServer::createGraspListMsg(const std::vector<Grasp>& hands)
{
  gpd::GraspConfigList msg;

  for (int i = 0; i < hands.size(); i++)
    msg.grasps.push_back(convertToGraspMsg(hands[i]));

  msg.header = cloud_camera_header_;

  return msg;
}


gpd::GraspConfig GraspDetectionServer::convertToGraspMsg(const Grasp& hand)
{
  gpd::GraspConfig msg;
  tf::pointEigenToMsg(hand.getGraspBottom(), msg.bottom);
  tf::pointEigenToMsg(hand.getGraspTop(), msg.top);
  tf::pointEigenToMsg(hand.getGraspSurface(), msg.surface);
  tf::vectorEigenToMsg(hand.getApproach(), msg.approach);
  tf::vectorEigenToMsg(hand.getBinormal(), msg.binormal);
  tf::vectorEigenToMsg(hand.getAxis(), msg.axis);
  msg.width.data = hand.getGraspWidth();
  msg.score.data = hand.getScore();
  tf::pointEigenToMsg(hand.getSample(), msg.sample);

  return msg;
}


int main(int argc, char** argv)
{
  // seed the random number generator
  std::srand(std::time(0));

  // initialize ROS
  ros::init(argc, argv, "detect_grasps_server");
  ros::NodeHandle node("~");

  GraspDetectionServer grasp_detection_server(node);

  ros::ServiceServer service = node.advertiseService("detect_grasps", &GraspDetectionServer::detectGrasps,
                                                     &grasp_detection_server);
  ROS_INFO("Grasp detection service is waiting for a point cloud ...");

  ros::spin();

  return 0;
}
