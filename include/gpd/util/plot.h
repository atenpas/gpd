/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Andreas ten Pas
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PLOT_H
#define PLOT_H

#include <pcl/visualization/pcl_visualizer.h>

#include <gpd/candidate/hand.h>
#include <gpd/candidate/hand_set.h>
#include <gpd/candidate/local_frame.h>
#include <gpd/descriptor/image_geometry.h>
#include <gpd/util/cloud.h>

namespace gpd {
namespace util {

typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;
typedef pcl::PointCloud<pcl::PointNormal> PointCloudPointNormal;

typedef boost::shared_ptr<pcl::visualization::PCLVisualizer> PCLVisualizer;

/**
 *
 * \brief Visualization utilities
 *
 * Provides visualization methods that use the PCL Visualizer. Allows to
 * visualize samples, surface normals, grasps, and point clouds.
 *
 */
class Plot {
 public:
  /**
   * \brief Constructor.
   * \param num_axes the number of orientation axes
   * \param num_orientations the number of hand orientations
   */
  Plot(int num_axes, int num_orientations)
      : num_axes_(num_axes), num_orientations_(num_orientations) {}

  void plotHandGeometry(const candidate::Hand &hand,
                        const PointCloudRGBA::Ptr &cloud,
                        const candidate::HandGeometry &hand_geom,
                        const descriptor::ImageGeometry &image_geom);

  /**
   * \brief Plot a list of hand sets and their associated volumes.
   * \param hand_set_list the list of grasp sets
   * \param cloud the point cloud to be plotted
   * \param str the title of the plot window
   * \param outer_diameter the outer diameter of the robot hand
   * \param finger_width the width of the robot fingers
   * \param hand_depth the depth of the robot hand
   * \param hand_height the height of the robot hand
   */
  void plotVolumes3D(
      const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
      const PointCloudRGBA::Ptr &cloud, std::string str, double outer_diameter,
      double finger_width, double hand_depth, double hand_height,
      double volume_width, double volume_depth, double volume_height);

  /**
   * \brief Plot a list of hands and their associated volumes.
   * \param hand_set_list the list of grasp sets
   * \param cloud the point cloud to be plotted
   * \param str the title of the plot window
   * \param outer_diameter the outer diameter of the robot hand
   * \param finger_width the width of the robot fingers
   * \param hand_depth the depth of the robot hand
   * \param hand_height the height of the robot hand
   */
  void plotVolumes3D(
      const std::vector<std::unique_ptr<candidate::Hand>> &hand_set_list,
      const PointCloudRGBA::Ptr &cloud, std::string str, double outer_diameter,
      double finger_width, double hand_depth, double hand_height,
      double volume_width, double volume_depth, double volume_height);

  void plotFingers3D(
      const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
      const PointCloudRGBA::Ptr &cloud, std::string str,
      const candidate::HandGeometry &geometry, bool draw_all = false,
      bool draw_frame = false);

  /**
   * \brief Plot a list of grasp sets with 3D cubes.
   * \param hand_set_list the list of grasp sets
   * \param cloud the point cloud to be plotted
   * \param str the title of the plot window
   * \param outer_diameter the outer diameter of the robot hand
   * \param finger_width the width of the robot fingers
   * \param hand_depth the depth of the robot hand
   * \param hand_height the height of the robot hand
   */
  void plotFingers3D(const std::vector<candidate::HandSet> &hand_set_list,
                     const PointCloudRGBA::Ptr &cloud, std::string str,
                     double outer_diameter, double finger_width,
                     double hand_depth, double hand_height,
                     bool draw_all = false, int num_axes = 1,
                     int num_orientations = 8);

  void plotFingers3D(
      const std::vector<std::unique_ptr<candidate::Hand>> &hand_list,
      const PointCloudRGBA::Ptr &cloud, const std::string &str,
      const candidate::HandGeometry &geometry, bool use_same_color = true);

  void plotAntipodalHands(
      const std::vector<std::unique_ptr<candidate::Hand>> &hand_list,
      const PointCloudRGBA::Ptr &cloud, const std::string &str,
      const candidate::HandGeometry &geometry);

  void plotValidHands(
      const std::vector<std::unique_ptr<candidate::Hand>> &hand_list,
      const PointCloudRGBA::Ptr &cloud, const PointCloudRGBA::Ptr &mesh,
      const std::string &str, const candidate::HandGeometry &geometry);

  /**
   * \brief Plot a list of grasps with 3D cubes.
   * \param hand_list the list of grasps
   * \param cloud the point cloud to be plotted
   * \param str the title of the plot window
   * \param outer_diameter the outer diameter of the robot hand
   * \param finger_width the width of the robot fingers
   * \param hand_depth the depth of the robot hand
   * \param hand_height the height of the robot hand
   */
  void plotFingers3D(const std::vector<candidate::Hand> &hand_list,
                     const PointCloudRGBA::Ptr &cloud, std::string str,
                     double outer_diameter, double finger_width,
                     double hand_depth, double hand_height,
                     bool draw_all = false);

  /**
   * \brief Plot a list of samples.
   * \param index_list the list of samples (indices into the point cloud)
   * \param cloud the point cloud to be plotted
   */
  void plotSamples(const std::vector<int> &index_list,
                   const PointCloudRGBA::Ptr &cloud);

  /**
   * \brief Plot a list of samples.
   * \param samples the list of samples (indices into the point cloud)
   * \param cloud the point cloud to be plotted
   */
  void plotSamples(const Eigen::Matrix3Xd &samples,
                   const PointCloudRGBA::Ptr &cloud);

  /**
   * \brief Plot a point cloud that contains samples.
   * \param samples_cloud the point cloud that contains the samples
   * \param cloud the point cloud to be plotted
   */
  void plotSamples(const PointCloudRGBA::Ptr &samples_cloud,
                   const PointCloudRGBA::Ptr &cloud);

  void plotNormals(const util::Cloud &cloud_cam, bool draw_camera_cone = false);

  void plotNormals(const PointCloudRGBA::Ptr &cloud,
                   const PointCloudRGBA::Ptr &cloud_samples,
                   const Eigen::Matrix3Xd &normals);

  /**
   * \brief Plot a list of normals.
   * \param cloud the point cloud to be plotted
   * \param normals the normals to be plotted
   */
  void plotNormals(const PointCloudRGBA::Ptr &cloud,
                   const Eigen::Matrix3Xd &normals);

  /**
   * \brief Plot a list of points and their normals.
   * \param pts the list of points to be plotted
   * \param normals the normals to be plotted
   */
  void plotNormals(const Eigen::Matrix3Xd &pts,
                   const Eigen::Matrix3Xd &normals);

  /**
   * \brief Plot a list of local reference frames.
   * \param frame_list the list of frames to be plotted
   * \param cloud the point cloud to be plotted
   */
  void plotLocalAxes(const std::vector<candidate::LocalFrame> &frames,
                     const PointCloudRGBA::Ptr &cloud);

  /**
   * \brief Plot the camera source for each point in the point cloud.
   * \param pts_cam_source_in the camera source for each point in the point
   * cloud
   * \param cloud the point cloud to be plotted
   */
  void plotCameraSource(const Eigen::VectorXi &pts_cam_source_in,
                        const PointCloudRGBA::Ptr &cloud);

  /**
   * \brief Plot a point cloud.
   * \param cloud_rgb the point cloud to be plotted
   * \param str the title of the plot window
   */
  void plotCloud(const PointCloudRGBA::Ptr &cloud_rgb,
                 const std::string &title);

 private:
  void addDimensions(const Eigen::Vector3d &center, const Eigen::Matrix3d &rot,
                     const Eigen::Vector3d &dimensions,
                     const Eigen::Matrix3d &colors,
                     const std::vector<std::string> &labels,
                     PCLVisualizer &viewer);

  void addDoubleArrow(const Eigen::Vector3d &start, const Eigen::Vector3d &end,
                      const std::string &label, const Eigen::Vector3d &rgb,
                      PCLVisualizer &viewer, bool is_label_at_start = false);

  void plotHand3D(PCLVisualizer &viewer, const candidate::Hand &hand,
                  const candidate::HandGeometry &geometry, int idx,
                  const Eigen::Vector3d &rgb);

  /**
   * \brief Plot a grasp.
   * \param viewer viewer the PCL visualizer in which the grasp is plotted
   * \param hand the grasp
   * \param outer_diameter the outer diameter of the robot hand
   * \param finger_width the width of the robot fingers
   * \param hand_depth the depth of the robot hand
   * \param hand_height the height of the robot hand
   * \param idx the ID of the grasp in the viewer
   */
  void plotHand3D(PCLVisualizer &viewer, const candidate::Hand &hand,
                  double outer_diameter, double finger_width, double hand_depth,
                  double hand_height, int idx, const Eigen::Vector3d &rgb);

  /**
   * \brief Plot a cube.
   * \param viewer viewer the PCL visualizer in which the grasp is plotted
   * \param position the center of the cube
   * \param rotation the orientation of the cube
   * \param width the width of the cube
   * \param height the height of the cube
   * \param depth the depth of the cube
   * \param name the name of the cube in the viewer
   */
  void plotCube(PCLVisualizer &viewer, const Eigen::Vector3d &position,
                const Eigen::Quaterniond &rotation, double width, double height,
                double depth, const std::string &name,
                const Eigen::Vector3d &rgb);

  void plotFrame(PCLVisualizer &viewer, const Eigen::Vector3d &translation,
                 const Eigen::Matrix3d &rotation, const std::string &id,
                 double axis_length = 0.02);
  /**
   * \brief Create a point cloud that stores the visual representations of the
   * grasps.
   * \param hand_list the list of grasps to be be stored in the point cloud
   * \param outer_diameter the outer diameter of the visual grasp representation
   */
  PointCloudRGBA::Ptr createFingersCloud(
      const std::vector<std::unique_ptr<candidate::Hand>> &hand_list,
      double outer_diameter);

  /**
   * \brief Convert an Eigen vector to a PCL point.
   * \param v the Eigen vector to be converted
   */
  pcl::PointXYZRGBA eigenVector3dToPointXYZRGBA(const Eigen::Vector3d &v);

  /**
   * \brief Add a point cloud with normals to a PCL visualizer.
   * \param viewer the PCL visualizer that the cloud is added to
   * \param cloud the cloud to be added
   * \param line_width the line width for drawing normals
   * \param color_cloud the color that is used to draw the cloud
   * \param color_normals the color that is used to draw the normals
   * \param cloud_name an identifier string for the cloud
   * \param normals_name an identifier string for the normals
   */
  void addCloudNormalsToViewer(PCLVisualizer &viewer,
                               const PointCloudPointNormal::Ptr &cloud,
                               double line_width, double *color_cloud,
                               double *color_normals,
                               const std::string &cloud_name,
                               const std::string &normals_name);

  /**
   * \brief Run/show a PCL visualizer until an escape key is hit.
   * \param viewer the PCL visualizer to be shown
   */
  void runViewer(PCLVisualizer &viewer);

  /**
   * \brief Create a PCL visualizer.
   * \param title the title of the visualization window
   */
  PCLVisualizer createViewer(std::string title);

  void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                             void *viewer_void);

  int num_orientations_;
  int num_axes_;
};

}  // namespace util
}  // namespace gpd

#endif /* PLOT_H */
