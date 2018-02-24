// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/filters/random_sample.h>
#include <pcl/point_types.h>

// ROS
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

// this project (services)
#include <gpd/detect_grasps.h>


typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;


int main(int argc, char* argv[])
{
  const std::string FRAME = "world";

  ros::init(argc, argv, "test_service");
  if (argc != 3)
  {
    ROS_INFO("Usage: test_service PATH_TO_PCD NUM_SAMPLES [VIEW_POINT]");
    return 1;
  }

  // Load point cloud from PCD file.
  PointCloud::Ptr cloud(new PointCloud);
  if (pcl::io::loadPCDFile<PointT> (argv[1], *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
  std::cout << "Loaded point cloud with " << cloud->width * cloud->height << " points.\n";

  // Setup a client for the grasp detection service.
  ros::NodeHandle nh;
  ros::ServiceClient client = nh.serviceClient<gpd::detect_grasps>("/detect_grasps_server/detect_grasps");

  // Draw uniform random samples from the point cloud.
  int num_samples = boost::lexical_cast<int>(argv[2]);
  std::vector<int> sample_indices;
  sample_indices.resize(num_samples);
  pcl::RandomSample<PointT> random_sample;
  random_sample.setInputCloud(cloud);
  random_sample.setSample(num_samples);
  random_sample.filter(sample_indices);

  // Create the service request.
  gpd::detect_grasps srv;

  // The point cloud indices used in the service request.
  srv.request.cloud_indexed.indices.resize(num_samples);
  for (int i=0; i < num_samples; i++)
  {
    srv.request.cloud_indexed.indices[i].data = sample_indices[i];
  }

  // The point cloud used in the service request.
  pcl::toROSMsg(*cloud, srv.request.cloud_indexed.cloud_sources.cloud);
  srv.request.cloud_indexed.cloud_sources.cloud.header.frame_id = FRAME;

  // The camera source for each point used in the service request.
  srv.request.cloud_indexed.cloud_sources.camera_source.resize(cloud->size());
  for (int i=0; i < cloud->size(); i++)
  {
    srv.request.cloud_indexed.cloud_sources.camera_source[i].data = 0;
  }

  // The view point used in the service request.
  srv.request.cloud_indexed.cloud_sources.view_points.resize(1);
  if (argc == 6)
  {
    srv.request.cloud_indexed.cloud_sources.view_points[0].x = boost::lexical_cast<double>(argv[3]);
    srv.request.cloud_indexed.cloud_sources.view_points[0].y = boost::lexical_cast<double>(argv[4]);
    srv.request.cloud_indexed.cloud_sources.view_points[0].z = boost::lexical_cast<double>(argv[5]);
  }
  else
  {
    srv.request.cloud_indexed.cloud_sources.view_points[0].x = 0;
    srv.request.cloud_indexed.cloud_sources.view_points[0].y = 0;
    srv.request.cloud_indexed.cloud_sources.view_points[0].z = 0;
  }

  if (client.call(srv))
  {
    ROS_INFO("# grasps: %d", (int) srv.response.grasp_configs.grasps.size());
    ROS_INFO_STREAM("frame: " << srv.response.grasp_configs.header.frame_id);
    for (int i=0; i < srv.response.grasp_configs.grasps.size(); i++)
    {
      ROS_INFO_STREAM(i << ": " << srv.response.grasp_configs.grasps[i].bottom.x << ", " <<
        srv.response.grasp_configs.grasps[i].bottom.y << ", " << srv.response.grasp_configs.grasps[i].bottom.z);
    }
  }
  else
  {
    ROS_ERROR("Failed to call service /detect_grasps_server/detect_grasps");
    return -1;
  }

  // Visualize the point cloud in rviz.
  ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("cloud_indexed", 1);
  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(*cloud, msg);
  msg.header.frame_id = FRAME;
  ros::Rate rate(1);
  while (ros::ok())
  {
      //publishing point cloud data
      cloud_pub.publish(msg);
      ros::spinOnce();
      rate.sleep();
  }

  return 0;
}
