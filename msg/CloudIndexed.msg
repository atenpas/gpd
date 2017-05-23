# This message holds a point cloud and a list of indices into the point cloud 
# at which to sample grasp candidates.

# The point cloud.
gpd/CloudSources cloud_sources

# The indices into the point cloud at which to sample grasp candidates.
std_msgs/Int64[] indices
