#include <gpd/grasp_plotter.h>


GraspPlotter::GraspPlotter(ros::NodeHandle& node, const HandSearch::Parameters& params)
{
  std::string rviz_topic;
  node.param("rviz_topic", rviz_topic, std::string(""));
  rviz_pub_ = node.advertise<visualization_msgs::MarkerArray>(rviz_topic, 1);

  hand_depth_ = params.hand_depth_;
  hand_height_ = params.hand_height_;
  outer_diameter_ = params.hand_outer_diameter_;
  finger_width_ = params.finger_width_;
}


void GraspPlotter::drawGrasps(const std::vector<Grasp>& hands, const std::string& frame)
{
  visualization_msgs::MarkerArray markers;
  markers = convertToVisualGraspMsg(hands, frame);
  rviz_pub_.publish(markers);
}


visualization_msgs::MarkerArray GraspPlotter::convertToVisualGraspMsg(const std::vector<Grasp>& hands,
  const std::string& frame_id)
{
  double hw = 0.5*outer_diameter_ - 0.5*finger_width_;

  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker left_finger, right_finger, base, approach;
  Eigen::Vector3d left_bottom, right_bottom, left_top, right_top, left_center, right_center, approach_center,
    base_center;

  for (int i = 0; i < hands.size(); i++)
  {
    left_bottom = hands[i].getGraspBottom() - hw * hands[i].getBinormal();
    right_bottom = hands[i].getGraspBottom() + hw * hands[i].getBinormal();
    left_top = left_bottom + hand_depth_ * hands[i].getApproach();
    right_top = right_bottom + hand_depth_ * hands[i].getApproach();
    left_center = left_bottom + 0.5*(left_top - left_bottom);
    right_center = right_bottom + 0.5*(right_top - right_bottom);
    base_center = left_bottom + 0.5*(right_bottom - left_bottom) - 0.01*hands[i].getApproach();
    approach_center = base_center - 0.04*hands[i].getApproach();

    Eigen::Vector3d finger_lwh, approach_lwh;
    finger_lwh << hand_depth_, finger_width_, hand_height_;
    approach_lwh << 0.08, finger_width_, hand_height_;

    base = createHandBaseMarker(left_bottom, right_bottom, hands[i].getFrame(), 0.02, hand_height_, i, frame_id);
    left_finger = createFingerMarker(left_center, hands[i].getFrame(), finger_lwh, i*3, frame_id);
    right_finger = createFingerMarker(right_center, hands[i].getFrame(), finger_lwh, i*3+1, frame_id);
    approach = createFingerMarker(approach_center, hands[i].getFrame(), approach_lwh, i*3+2, frame_id);

    marker_array.markers.push_back(left_finger);
    marker_array.markers.push_back(right_finger);
    marker_array.markers.push_back(approach);
    marker_array.markers.push_back(base);
  }

  return marker_array;
}


visualization_msgs::Marker GraspPlotter::createFingerMarker(const Eigen::Vector3d& center,
  const Eigen::Matrix3d& frame, const Eigen::Vector3d& lwh, int id, const std::string& frame_id)
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
  marker.scale.x = lwh(0); // forward direction
  marker.scale.y = lwh(1); // hand closing direction
  marker.scale.z = lwh(2); // hand vertical direction

  marker.color.a = 0.5;
  marker.color.r = 0.0;
  marker.color.g = 0.0;
  marker.color.b = 0.5;

  return marker;
}


visualization_msgs::Marker GraspPlotter::createHandBaseMarker(const Eigen::Vector3d& start,
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


