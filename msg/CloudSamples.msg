# This message holds a point cloud and a list of samples at which the grasp 
# detector should search for grasp candidates.

# The point cloud.
gpd/CloudSources cloud_sources

# The samples, as (x,y,z) points, at which to search for grasp candidates. 
geometry_msgs/Point[] samples
