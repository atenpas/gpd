#include "../../../gpd/include/nodes/grasp_detection_node.h"


/** constants for input point cloud types */
const int GraspDetectionNode::POINT_CLOUD_2 = 0; ///< sensor_msgs/PointCloud2
const int GraspDetectionNode::CLOUD_INDEXED = 1; ///< cloud with indices
const int GraspDetectionNode::CLOUD_SAMPLES = 2; ///< cloud with (x,y,z) samples


GraspDetectionNode::GraspDetectionNode(ros::NodeHandle& node) : has_cloud_(false), has_normals_(false),
  size_left_cloud_(0), has_samples_(true), frame_("")
{
  cloud_camera_ = NULL;

  nh_ = node; // Assign the NodeHandle to the private variable

  // set camera viewpoint to default origin
  std::vector<double> camera_position;
  nh_.getParam("camera_position", camera_position);
  view_point_ << camera_position[0], camera_position[1], camera_position[2];

  // choose sampling method for grasp detection
  nh_.param("use_importance_sampling", use_importance_sampling_, false);

  if (use_importance_sampling_)
  {
    importance_sampling_ = new SequentialImportanceSampling(nh_);
  }
  grasp_detector_ = new GraspDetector(nh_);

  // Read input cloud and sample ROS topics parameters.
  int cloud_type;
  nh_.param("cloud_type", cloud_type, POINT_CLOUD_2);
  std::string cloud_topic;
  nh_.param("cloud_topic", cloud_topic, std::string("/camera/depth_registered/points"));
  std::string samples_topic;
  nh_.param("samples_topic", samples_topic, std::string(""));
  std::string rviz_topic;
  nh_.param("rviz_topic", rviz_topic, std::string(""));

  if (!rviz_topic.empty())
  {
    grasps_rviz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(rviz_topic, 1);
    use_rviz_ = true;
  }
  else
  {
    use_rviz_ = false;
  }

  // subscribe to input point cloud ROS topic
  if (cloud_type == POINT_CLOUD_2)
    cloud_sub_ = nh_.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_callback, this);
  else if (cloud_type == CLOUD_INDEXED)
    cloud_sub_ = nh_.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_indexed_callback, this);
  else if (cloud_type == CLOUD_SAMPLES)
  {
    cloud_sub_ = nh_.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_samples_callback, this);
    //    grasp_detector_->setUseIncomingSamples(true);
    has_samples_ = false;
  }

  // subscribe to input samples ROS topic
  if (!samples_topic.empty())
  {
    samples_sub_ = nh_.subscribe(samples_topic, 1, &GraspDetectionNode::samples_callback, this);
    has_samples_ = false;
  }

  // uses ROS topics to publish grasp candidates, antipodal grasps, and grasps after clustering
  grasps_pub_ = nh_.advertise<gpd::GraspConfigList>("clustered_grasps", 10);

  // Advertise the SetParameters service
  srv_set_params_ = nh_.advertiseService("/gpd/set_params", &GraspDetectionNode::set_params_callback, this);

  nh_.getParam("workspace", workspace_);
}


void GraspDetectionNode::run()
{
  ros::Rate rate(100);
  ROS_INFO("Waiting for point cloud to arrive ...");

  while (ros::ok())
  {
    if (has_cloud_)
    {
      // detect grasps in point cloud
      std::vector<Grasp> grasps = detectGraspPosesInTopic();

      // visualize grasps in rviz
      if (use_rviz_)
      {
        const HandSearch::Parameters& params = grasp_detector_->getHandSearchParameters();
        grasps_rviz_pub_.publish(convertToVisualGraspMsg(grasps, params.hand_outer_diameter_, params.hand_depth_,
                                                         params.finger_width_, params.hand_height_, frame_));
      }

      // reset the system
      has_cloud_ = false;
      has_samples_ = false;
      has_normals_ = false;
      ROS_INFO("Waiting for point cloud to arrive ...");
    }

    ros::spinOnce();
    rate.sleep();
  }
}

bool GraspDetectionNode::set_params_callback(gpd::SetParameters::Request &req, gpd::SetParameters::Response &resp)
{
  // Delete the existing sampler and detectors
  if (use_importance_sampling_)
  {
    delete importance_sampling_;
  }
  delete grasp_detector_;

  // Set the workspace from the request
  if (req.set_workspace)
  {
    workspace_.clear();
    for (int i = 0; i < req.workspace.size(); i++){

      workspace_.push_back(req.workspace[i]);
    }
    nh_.setParam("workspace", workspace_);
  }

  // Set the workspace_grasps from the request
  if (req.set_workspace_grasps)
  {
    nh_.setParam("filter_grasps", true);
    std::vector<double> workspace_grasps;
    for (int i = 0; i < req.workspace_grasps.size(); i++)
    {
      workspace_grasps.push_back(req.workspace_grasps[i]);
    }
    nh_.setParam("workspace_grasps", workspace_grasps);
  }
  else
  {
    nh_.setParam("filter_grasps", false);
  }

  if (req.set_camera_position)
  {
    view_point_ << req.camera_position[0], req.camera_position[1], req.camera_position[2];
    std::vector<double> camera_position;
    camera_position.push_back(view_point_.x());
    camera_position.push_back(view_point_.y());
    camera_position.push_back(view_point_.z());
    nh_.setParam("camera_position", camera_position);
  }

  // Creating new sampler and detector so they load the new rosparams
  if (use_importance_sampling_)
  {
    importance_sampling_ = new SequentialImportanceSampling(nh_);
  }
  grasp_detector_ = new GraspDetector(nh_);

  resp.success = true;
  return true;
}

std::vector<Grasp> GraspDetectionNode::detectGraspPosesInTopic()
{
  // detect grasp poses
  std::vector<Grasp> grasps;

  if (use_importance_sampling_)
  {
    cloud_camera_->filterWorkspace(workspace_);
    cloud_camera_->voxelizeCloud(0.003);
    cloud_camera_->calculateNormals(4);
    grasps = importance_sampling_->detectGrasps(*cloud_camera_);
  }
  else
  {
    // preprocess the point cloud
    grasp_detector_->preprocessPointCloud(*cloud_camera_);

    // detect grasps in the point cloud
    grasps = grasp_detector_->detectGrasps(*cloud_camera_);
  }

  // Publish the selected grasps.
  gpd::GraspConfigList selected_grasps_msg = createGraspListMsg(grasps);
  grasps_pub_.publish(selected_grasps_msg);
  ROS_INFO_STREAM("Published " << selected_grasps_msg.grasps.size() << " highest-scoring grasps.");

  return grasps;
}


std::vector<int> GraspDetectionNode::getSamplesInBall(const PointCloudRGBA::Ptr& cloud,
  const pcl::PointXYZRGBA& centroid, float radius)
{
  std::vector<int> indices;
  std::vector<float> dists;
  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud);
  kdtree.radiusSearch(centroid, radius, indices, dists);
  return indices;
}


void GraspDetectionNode::cloud_callback(const sensor_msgs::PointCloud2& msg)
{
  if (!has_cloud_)
  {
    delete cloud_camera_;
    cloud_camera_ = NULL;

    Eigen::Matrix3Xd view_points(3,1);
    view_points.col(0) = view_point_;

    if (msg.fields.size() == 6 && msg.fields[3].name == "normal_x" && msg.fields[4].name == "normal_y"
      && msg.fields[5].name == "normal_z")
    {
      PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
      pcl::fromROSMsg(msg, *cloud);
      cloud_camera_ = new CloudCamera(cloud, 0, view_points);
      cloud_camera_header_ = msg.header;
      ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points and normals.");
    }
    else
    {
      PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
      pcl::fromROSMsg(msg, *cloud);
      cloud_camera_ = new CloudCamera(cloud, 0, view_points);
      cloud_camera_header_ = msg.header;
      ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points.");
    }

    has_cloud_ = true;
    frame_ = msg.header.frame_id;
  }
}


void GraspDetectionNode::cloud_indexed_callback(const gpd::CloudIndexed& msg)
{
  if (!has_cloud_)
  {
    initCloudCamera(msg.cloud_sources);

    // Set the indices at which to sample grasp candidates.
    std::vector<int> indices(msg.indices.size());
    for (int i=0; i < indices.size(); i++)
    {
      indices[i] = msg.indices[i].data;
    }
    cloud_camera_->setSampleIndices(indices);

    has_cloud_ = true;
    frame_ = msg.cloud_sources.cloud.header.frame_id;

    ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points, and "
      << msg.indices.size() << " samples");
  }
}


void GraspDetectionNode::cloud_samples_callback(const gpd::CloudSamples& msg)
{
  if (!has_cloud_)
  {
    initCloudCamera(msg.cloud_sources);

    // Set the samples at which to sample grasp candidates.
    Eigen::Matrix3Xd samples(3, msg.samples.size());
    for (int i=0; i < msg.samples.size(); i++)
    {
      samples.col(i) << msg.samples[i].x, msg.samples[i].y, msg.samples[i].z;
    }
    cloud_camera_->setSamples(samples);

    has_cloud_ = true;
    has_samples_ = true;
    frame_ = msg.cloud_sources.cloud.header.frame_id;

    ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points, and "
      << cloud_camera_->getSamples().cols() << " samples");
  }
}


void GraspDetectionNode::samples_callback(const gpd::SamplesMsg& msg)
{
  if (!has_samples_)
  {
    Eigen::Matrix3Xd samples(3, msg.samples.size());

    for (int i=0; i < msg.samples.size(); i++)
    {
      samples.col(i) << msg.samples[i].x, msg.samples[i].y, msg.samples[i].z;
    }

    cloud_camera_->setSamples(samples);
    has_samples_ = true;

    ROS_INFO_STREAM("Received grasp samples message with " << msg.samples.size() << " samples");
  }
}


void GraspDetectionNode::initCloudCamera(const gpd::CloudSources& msg)
{
  // clean up
  delete cloud_camera_;
  cloud_camera_ = NULL;

  // Set view points.
  Eigen::Matrix3Xd view_points(3, msg.view_points.size());
  for (int i = 0; i < msg.view_points.size(); i++)
  {
    view_points.col(i) << msg.view_points[i].x, msg.view_points[i].y, msg.view_points[i].z;
  }

  // Set point cloud.
  if (msg.cloud.fields.size() == 6 && msg.cloud.fields[3].name == "normal_x"
    && msg.cloud.fields[4].name == "normal_y" && msg.cloud.fields[5].name == "normal_z")
  {
    PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
    pcl::fromROSMsg(msg.cloud, *cloud);

    // TODO: multiple cameras can see the same point
    Eigen::MatrixXi camera_source = Eigen::MatrixXi::Zero(view_points.cols(), cloud->size());
    for (int i = 0; i < msg.camera_source.size(); i++)
    {
      camera_source(msg.camera_source[i].data, i) = 1;
    }

    cloud_camera_ = new CloudCamera(cloud, camera_source, view_points);
  }
  else
  {
    PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
    pcl::fromROSMsg(msg.cloud, *cloud);

    // TODO: multiple cameras can see the same point
    Eigen::MatrixXi camera_source = Eigen::MatrixXi::Zero(view_points.cols(), cloud->size());
    for (int i = 0; i < msg.camera_source.size(); i++)
    {
      camera_source(msg.camera_source[i].data, i) = 1;
    }

    cloud_camera_ = new CloudCamera(cloud, camera_source, view_points);
    std::cout << "view_points:\n" << view_points << "\n";
  }
}


gpd::GraspConfigList GraspDetectionNode::createGraspListMsg(const std::vector<Grasp>& hands)
{
  gpd::GraspConfigList msg;

  for (int i = 0; i < hands.size(); i++)
    msg.grasps.push_back(convertToGraspMsg(hands[i]));

  msg.header = cloud_camera_header_;

  return msg;
}


gpd::GraspConfig GraspDetectionNode::convertToGraspMsg(const Grasp& hand)
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


visualization_msgs::MarkerArray GraspDetectionNode::convertToVisualGraspMsg(const std::vector<Grasp>& hands,
  double outer_diameter, double hand_depth, double finger_width, double hand_height, const std::string& frame_id)
{
  double width = outer_diameter;
  double hw = 0.5 * width;

  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker left_finger, right_finger, base, approach;
  Eigen::Vector3d left_bottom, right_bottom, left_top, right_top, left_center, right_center, approach_center,
    base_center;

  for (int i = 0; i < hands.size(); i++)
  {
    left_bottom = hands[i].getGraspBottom() - (hw - 0.5*finger_width) * hands[i].getBinormal();
    right_bottom = hands[i].getGraspBottom() + (hw - 0.5*finger_width) * hands[i].getBinormal();
    left_top = left_bottom + hand_depth * hands[i].getApproach();
    right_top = right_bottom + hand_depth * hands[i].getApproach();
    left_center = left_bottom + 0.5*(left_top - left_bottom);
    right_center = right_bottom + 0.5*(right_top - right_bottom);
    base_center = left_bottom + 0.5*(right_bottom - left_bottom) - 0.01*hands[i].getApproach();
    approach_center = base_center - 0.04*hands[i].getApproach();

    base = createHandBaseMarker(left_bottom, right_bottom, hands[i].getFrame(), 0.02, hand_height, i, frame_id);
    left_finger = createFingerMarker(left_center, hands[i].getFrame(), hand_depth, finger_width, hand_height, i*3, frame_id);
    right_finger = createFingerMarker(right_center, hands[i].getFrame(), hand_depth, finger_width, hand_height, i*3+1, frame_id);
    approach = createFingerMarker(approach_center, hands[i].getFrame(), 0.08, finger_width, hand_height, i*3+2, frame_id);

    marker_array.markers.push_back(left_finger);
    marker_array.markers.push_back(right_finger);
    marker_array.markers.push_back(approach);
    marker_array.markers.push_back(base);
  }

  return marker_array;
}


visualization_msgs::Marker GraspDetectionNode::createFingerMarker(const Eigen::Vector3d& center,
  const Eigen::Matrix3d& frame, double length, double width, double height, int id, const std::string& frame_id)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time();
  marker.ns = "finger";
  marker.id = id;
  marker.type = visualization_msgs::Marker::CUBE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = center(0);
  marker.pose.position.y = center(1);
  marker.pose.position.z = center(2);
  marker.lifetime = ros::Duration(10);

  // use orientation of hand frame
  Eigen::Quaterniond quat(frame);
  marker.pose.orientation.x = quat.x();
  marker.pose.orientation.y = quat.y();
  marker.pose.orientation.z = quat.z();
  marker.pose.orientation.w = quat.w();

  // these scales are relative to the hand frame (unit: meters)
  marker.scale.x = length; // forward direction
  marker.scale.y = width; // hand closing direction
  marker.scale.z = height; // hand vertical direction

  marker.color.a = 0.5;
  marker.color.r = 0.0;
  marker.color.g = 0.0;
  marker.color.b = 0.5;

  return marker;
}


visualization_msgs::Marker GraspDetectionNode::createHandBaseMarker(const Eigen::Vector3d& start,
  const Eigen::Vector3d& end, const Eigen::Matrix3d& frame, double length, double height, int id,
  const std::string& frame_id)
{
  Eigen::Vector3d center = start + 0.5 * (end - start);

  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time();
  marker.ns = "hand_base";
  marker.id = id;
  marker.type = visualization_msgs::Marker::CUBE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = center(0);
  marker.pose.position.y = center(1);
  marker.pose.position.z = center(2);
  marker.lifetime = ros::Duration(10);

  // use orientation of hand frame
  Eigen::Quaterniond quat(frame);
  marker.pose.orientation.x = quat.x();
  marker.pose.orientation.y = quat.y();
  marker.pose.orientation.z = quat.z();
  marker.pose.orientation.w = quat.w();

  // these scales are relative to the hand frame (unit: meters)
  marker.scale.x = length; // forward direction
  marker.scale.y = (end - start).norm(); // hand closing direction
  marker.scale.z = height; // hand vertical direction

  marker.color.a = 0.5;
  marker.color.r = 0.0;
  marker.color.g = 0.0;
  marker.color.b = 1.0;

  return marker;
}


int main(int argc, char** argv)
{
  // seed the random number generator
  std::srand(std::time(0));

  // initialize ROS
  ros::init(argc, argv, "detect_grasps");
  ros::NodeHandle node("~");

  GraspDetectionNode grasp_detection(node);
  grasp_detection.run();

  return 0;
}
