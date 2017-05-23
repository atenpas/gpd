# This message holds a point cloud that can be a combination of point clouds 
# from different camera sources (at least one). For each point in the cloud, 
# this message also stores the index of the camera that produced the point.

# The point cloud.
sensor_msgs/PointCloud2 cloud

# For each point in the cloud, the index of the camera that acquired the point.
std_msgs/Int64[] camera_source

# A list of camera positions at which the point cloud was acquired.
geometry_msgs/Point[] view_points