#include <gpd/util/plot.h>

#include <chrono>
#include <thread>

namespace gpd {
namespace util {

void Plot::plotHandGeometry(const candidate::Hand &hand,
                            const PointCloudRGBA::Ptr &cloud,
                            const candidate::HandGeometry &hand_geom,
                            const descriptor::ImageGeometry &image_geom) {
  Eigen::Vector3d vol_rgb(0.0, 0.8, 0.0);
  Eigen::Vector3d hand_rgb(0.0, 0.5, 0.5);
  PCLVisualizer viewer = createViewer("Hand Geometry");
  plotHand3D(viewer, hand, hand_geom, 0, hand_rgb);
  Eigen::Vector3d vol_pos =
      hand.getPosition() + 0.5 * image_geom.depth_ * hand.getApproach();
  Eigen::Quaterniond vol_quat(hand.getFrame());
  plotCube(viewer, vol_pos, vol_quat, image_geom.depth_,
           image_geom.outer_diameter_, 2.0 * image_geom.height_, "volume",
           vol_rgb);

  Eigen::Vector3d dimensions(hand_geom.depth_, hand_geom.outer_diameter_,
                             2.0 * hand_geom.height_);
  Eigen::Matrix3d colors;
  colors << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  Eigen::Vector3d center = hand.getPosition() -
                           hand_geom.height_ * hand.getAxis() -
                           0.5 * hand_geom.outer_diameter_ * hand.getBinormal();
  std::vector<std::string> labels;
  labels.push_back("hand_depth");
  labels.push_back("hand_outer_diameter");
  labels.push_back("hand_height * 2");
  addDimensions(center, hand.getOrientation(), dimensions, colors, labels,
                viewer);

  Eigen::Vector3d p = center + 2.0 * hand_geom.height_ * hand.getAxis();
  Eigen::Vector3d q = p + hand_geom.finger_width_ * hand.getBinormal();
  addDoubleArrow(p, q, "finger_width", Eigen::Vector3d(0.0, 1.0, 1.0), viewer);

  dimensions << image_geom.depth_, -image_geom.outer_diameter_,
      2.0 * image_geom.height_;
  colors *= 0.6;
  center = hand.getPosition() - image_geom.height_ * hand.getAxis() +
           0.5 * image_geom.outer_diameter_ * hand.getBinormal();
  labels.resize(0);
  labels.push_back("volume_depth");
  labels.push_back("volume_width");
  labels.push_back("volume_height * 2");
  addDimensions(center, hand.getOrientation(), dimensions, colors, labels,
                viewer);

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(
      cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

  runViewer(viewer);
}

void Plot::addDimensions(const Eigen::Vector3d &center,
                         const Eigen::Matrix3d &rot,
                         const Eigen::Vector3d &dimensions,
                         const Eigen::Matrix3d &colors,
                         const std::vector<std::string> &labels,
                         PCLVisualizer &viewer) {
  bool is_label_at_start[3] = {false, true, false};
  for (size_t i = 0; i < 3; i++) {
    Eigen::Vector3d p = center;
    Eigen::Vector3d q = p + dimensions(i) * rot.col(i);
    addDoubleArrow(p, q, labels[i], colors.row(i), viewer,
                   is_label_at_start[i]);
  }
}

void Plot::addDoubleArrow(const Eigen::Vector3d &start,
                          const Eigen::Vector3d &end, const std::string &label,
                          const Eigen::Vector3d &rgb, PCLVisualizer &viewer,
                          bool is_label_at_start) {
  pcl::PointXYZRGB p;
  pcl::PointXYZRGB q;
  p.getVector3fMap() = start.cast<float>();
  q.getVector3fMap() = end.cast<float>();
  viewer->addArrow(p, q, rgb[0], rgb[1], rgb[2], false,
                   label + std::to_string(0));
  viewer->addArrow(q, p, rgb[0], rgb[1], rgb[2], false,
                   label + std::to_string(1));
  if (is_label_at_start) {
    viewer->addText3D(label, p, 0.008, rgb[0], rgb[1], rgb[2]);
  } else {
    viewer->addText3D(label, q, 0.008, rgb[0], rgb[1], rgb[2]);
  }
}

void Plot::plotVolumes3D(
    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    const PointCloudRGBA::Ptr &cloud, std::string str, double outer_diameter,
    double finger_width, double hand_depth, double hand_height,
    double volume_width, double volume_depth, double volume_height) {
  Eigen::Vector3d vol_rgb(0.0, 0.8, 0.0);
  Eigen::Vector3d hand_rgb(0.0, 0.5, 0.5);

  PCLVisualizer viewer = createViewer(str);

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<std::unique_ptr<candidate::Hand>> &hands =
        hand_set_list[i]->getHands();

    for (int j = 0; j < hands.size(); j++) {
      if (!hand_set_list[i]->getIsValid()[j]) {
        continue;
      }

      int idx = i * hands.size() + j;

      // Plot the hand.
      plotHand3D(viewer, *hands[j], outer_diameter, finger_width, hand_depth,
                 hand_height, idx, hand_rgb);

      // Plot the associated volume.
      Eigen::Vector3d vol_pos = hands[j]->getPosition() +
                                0.5 * volume_depth * hands[j]->getApproach();
      Eigen::Quaterniond vol_quat(hands[i]->getFrame());
      std::string num = std::to_string(idx);
      plotCube(viewer, vol_pos, vol_quat, volume_depth, volume_width,
               volume_height, "volume_" + num, vol_rgb);
    }
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(
      cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

  runViewer(viewer);
}

void Plot::plotVolumes3D(
    const std::vector<std::unique_ptr<candidate::Hand>> &hand_list,
    const PointCloudRGBA::Ptr &cloud, std::string str, double outer_diameter,
    double finger_width, double hand_depth, double hand_height,
    double volume_width, double volume_depth, double volume_height) {
  Eigen::Vector3d vol_rgb(0.0, 0.8, 0.0);
  Eigen::Vector3d hand_rgb(0.0, 0.5, 0.5);

  PCLVisualizer viewer = createViewer(str);

  for (int i = 0; i < hand_list.size(); i++) {
    // Plot the hand.
    plotHand3D(viewer, *hand_list[i], outer_diameter, finger_width, hand_depth,
               hand_height, i, hand_rgb);

    // Plot the associated volume.
    Eigen::Vector3d vol_pos = hand_list[i]->getPosition() +
                              0.5 * volume_depth * hand_list[i]->getApproach();
    Eigen::Quaterniond vol_quat(hand_list[i]->getFrame());
    std::string num = std::to_string(i);
    plotCube(viewer, vol_pos, vol_quat, volume_depth, volume_width,
             volume_height, "volume_" + num, vol_rgb);
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(
      cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

  runViewer(viewer);
}

void Plot::plotFingers3D(
    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    const PointCloudRGBA::Ptr &cloud, std::string str,
    const candidate::HandGeometry &geometry, bool draw_all, bool draw_frame) {
  const Eigen::Vector3d RGB[3] = {Eigen::Vector3d(0.5, 0, 0),
                                  Eigen::Vector3d(0, 0.5, 0),
                                  Eigen::Vector3d(0, 0, 0.5)};
  PCLVisualizer viewer = createViewer(str);
  const int max_hands_per_set_ = num_axes_ * num_orientations_;

  for (int i = 0; i < hand_set_list.size(); i++) {
    for (int j = 0; j < hand_set_list[i]->getHands().size(); j++) {
      if (draw_all || hand_set_list[i]->getIsValid()(j)) {
        // Choose color based on rotation axis.
        Eigen::Vector3d rgb;
        if (draw_all) {
          int idx_color = j / num_orientations_;
          rgb = RGB[idx_color];
        } else {
          rgb << 0.0, 0.5, 0.5;
        }
        plotHand3D(viewer, *hand_set_list[i]->getHands()[j], geometry,
                   i * max_hands_per_set_ + j, rgb);
      }
    }

    if (draw_frame) {
      plotFrame(viewer, hand_set_list[i]->getSample(),
                hand_set_list[i]->getFrame(), std::to_string(i));
    }
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(
      cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

  runViewer(viewer);
}

void Plot::plotFingers3D(
    const std::vector<std::unique_ptr<candidate::Hand>> &hand_list,
    const PointCloudRGBA::Ptr &cloud, const std::string &str,
    const candidate::HandGeometry &geometry, bool use_same_color) {
  PCLVisualizer viewer = createViewer(str);

  double min = std::numeric_limits<float>::max();
  double max = std::numeric_limits<float>::min();
  for (int i = 0; i < hand_list.size(); i++) {
    if (hand_list[i]->getScore() < min) {
      min = hand_list[i]->getScore();
    }
    if (hand_list[i]->getScore() > max) {
      max = hand_list[i]->getScore();
    }
  }

  Eigen::Vector3d hand_rgb;
  if (use_same_color) {
    hand_rgb << 0.0, 0.5, 0.5;
  }

  for (int i = 0; i < hand_list.size(); i++) {
    if (!use_same_color) {
      double c = (hand_list[i]->getScore() - min) / (max - min);
      hand_rgb = Eigen::Vector3d(1.0 - c, c, 0.0);
    }
    plotHand3D(viewer, *hand_list[i], geometry, i, hand_rgb);
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(
      cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

  runViewer(viewer);
}

void Plot::plotAntipodalHands(
    const std::vector<std::unique_ptr<candidate::Hand>> &hand_list,
    const PointCloudRGBA::Ptr &cloud, const std::string &str,
    const candidate::HandGeometry &geometry) {
  PCLVisualizer viewer = createViewer(str);

  Eigen::Vector3d antipodal_color;
  Eigen::Vector3d non_antipodal_color;
  antipodal_color << 0.0, 0.7, 0.0;
  non_antipodal_color << 0.7, 0.0, 0.0;

  for (int i = 0; i < hand_list.size(); i++) {
    Eigen::Vector3d color =
        hand_list[i]->isFullAntipodal() ? antipodal_color : non_antipodal_color;
    plotHand3D(viewer, *hand_list[i], geometry, i, color);
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(
      cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

  runViewer(viewer);
}

void Plot::plotValidHands(
    const std::vector<std::unique_ptr<candidate::Hand>> &hand_list,
    const PointCloudRGBA::Ptr &cloud, const PointCloudRGBA::Ptr &mesh,
    const std::string &str, const candidate::HandGeometry &geometry) {
  PCLVisualizer viewer = createViewer(str);

  Eigen::Vector3d antipodal_color;
  Eigen::Vector3d non_antipodal_color;
  antipodal_color << 0.0, 0.7, 0.0;
  non_antipodal_color << 0.7, 0.0, 0.0;

  for (int i = 0; i < hand_list.size(); i++) {
    Eigen::Vector3d color =
        hand_list[i]->isFullAntipodal() ? antipodal_color : non_antipodal_color;
    plotHand3D(viewer, *hand_list[i], geometry, i, color);
  }

  viewer->addPointCloud<pcl::PointXYZRGBA>(mesh, "mesh");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "mesh");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "mesh");

  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_OPACITY, 0.6, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

  runViewer(viewer);
}

void Plot::plotFingers3D(const std::vector<candidate::HandSet> &hand_set_list,
                         const PointCloudRGBA::Ptr &cloud, std::string str,
                         double outer_diameter, double finger_width,
                         double hand_depth, double hand_height, bool draw_all,
                         int num_axes, int num_orientations) {
  const Eigen::Vector3d RGB[3] = {Eigen::Vector3d(0.5, 0, 0),
                                  Eigen::Vector3d(0, 0.5, 0),
                                  Eigen::Vector3d(0, 0, 0.5)};

  PCLVisualizer viewer = createViewer(str);

  for (int i = 0; i < hand_set_list.size(); i++) {
    for (int j = 0; j < hand_set_list[i].getHands().size(); j++) {
      if (draw_all || hand_set_list[i].getIsValid()(j)) {
        Eigen::Vector3d rgb;
        if (draw_all) {
          int idx_color = j / num_orientations;
          rgb = RGB[idx_color];
        } else {
          rgb << 0.0, 0.5, 0.5;
        }
        plotHand3D(viewer, *hand_set_list[i].getHands()[j], outer_diameter,
                   finger_width, hand_depth, hand_height,
                   i * (num_axes * num_orientations) + j, rgb);
      }
    }
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(
      cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

  runViewer(viewer);
}

void Plot::plotFingers3D(const std::vector<candidate::Hand> &hand_list,
                         const PointCloudRGBA::Ptr &cloud, std::string str,
                         double outer_diameter, double finger_width,
                         double hand_depth, double hand_height, bool draw_all) {
  PCLVisualizer viewer = createViewer(str);
  Eigen::Vector3d hand_rgb(0.0, 0.5, 0.5);

  for (int i = 0; i < hand_list.size(); i++) {
    plotHand3D(viewer, hand_list[i], outer_diameter, finger_width, hand_depth,
               hand_height, i, hand_rgb);
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(
      cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

  runViewer(viewer);
}

void Plot::plotHand3D(PCLVisualizer &viewer, const candidate::Hand &hand,
                      const candidate::HandGeometry &geometry, int idx,
                      const Eigen::Vector3d &rgb) {
  plotHand3D(viewer, hand, geometry.outer_diameter_, geometry.finger_width_,
             geometry.depth_, geometry.height_, idx, rgb);
}

void Plot::plotHand3D(PCLVisualizer &viewer, const candidate::Hand &hand,
                      double outer_diameter, double finger_width,
                      double hand_depth, double hand_height, int idx,
                      const Eigen::Vector3d &rgb) {
  const double hw = 0.5 * outer_diameter;
  const double base_depth = 0.02;
  const double approach_depth = 0.07;

  Eigen::Vector3d left_bottom =
      hand.getPosition() - (hw - 0.5 * finger_width) * hand.getBinormal();
  Eigen::Vector3d right_bottom =
      hand.getPosition() + (hw - 0.5 * finger_width) * hand.getBinormal();
  Eigen::VectorXd left_center =
      left_bottom + 0.5 * hand_depth * hand.getApproach();
  Eigen::VectorXd right_center =
      right_bottom + 0.5 * hand_depth * hand.getApproach();
  Eigen::Vector3d base_center = left_bottom +
                                0.5 * (right_bottom - left_bottom) -
                                0.01 * hand.getApproach();
  Eigen::Vector3d approach_center = base_center - 0.04 * hand.getApproach();

  const Eigen::Quaterniond quat(hand.getFrame());
  const std::string num = std::to_string(idx);

  plotCube(viewer, left_center, quat, hand_depth, finger_width, hand_height,
           "left_finger_" + num, rgb);
  plotCube(viewer, right_center, quat, hand_depth, finger_width, hand_height,
           "right_finger_" + num, rgb);
  plotCube(viewer, base_center, quat, base_depth, outer_diameter, hand_height,
           "base_" + num, rgb);
  plotCube(viewer, approach_center, quat, approach_depth, finger_width,
           0.5 * hand_height, "approach_" + num, rgb);
}

void Plot::plotCube(PCLVisualizer &viewer, const Eigen::Vector3d &position,
                    const Eigen::Quaterniond &rotation, double width,
                    double height, double depth, const std::string &name,
                    const Eigen::Vector3d &rgb) {
  viewer->addCube(position.cast<float>(), rotation.cast<float>(), width, height,
                  depth, name);
  viewer->setShapeRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
      pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, name);
  viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                      rgb(0), rgb(1), rgb(2), name);
  viewer->setShapeRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_OPACITY, 0.25, name);
}

void Plot::plotFrame(PCLVisualizer &viewer, const Eigen::Vector3d &translation,
                     const Eigen::Matrix3d &rotation, const std::string &id,
                     double axis_length) {
  const Eigen::Matrix3d pts = axis_length * rotation;
  const std::string names[3] = {"normal_" + id, "binormal_" + id,
                                "curvature_" + id};
  const double colors[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  pcl::PointXYZ p;
  p.getVector3fMap() = translation.cast<float>();
  for (int i = 0; i < 3; i++) {
    pcl::PointXYZ q;
    q.getVector3fMap() = (translation + pts.col(i)).cast<float>();
    viewer->addLine<pcl::PointXYZ>(p, q, colors[i][0], colors[i][1],
                                   colors[i][2], names[i]);
    viewer->setShapeRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, names[i]);
  }
}

void Plot::plotSamples(const std::vector<int> &index_list,
                       const PointCloudRGBA::Ptr &cloud) {
  PointCloudRGBA::Ptr samples_cloud(new PointCloudRGBA);
  for (int i = 0; i < index_list.size(); i++) {
    samples_cloud->points.push_back(cloud->points[index_list[i]]);
  }

  plotSamples(samples_cloud, cloud);
}

void Plot::plotSamples(const Eigen::Matrix3Xd &samples,
                       const PointCloudRGBA::Ptr &cloud) {
  PointCloudRGBA::Ptr samples_cloud(new PointCloudRGBA);
  for (int i = 0; i < samples.cols(); i++) {
    pcl::PointXYZRGBA p;
    p.x = samples.col(i)(0);
    p.y = samples.col(i)(1);
    p.z = samples.col(i)(2);
    samples_cloud->points.push_back(p);
  }

  plotSamples(samples_cloud, cloud);
}

void Plot::plotSamples(const PointCloudRGBA::Ptr &samples_cloud,
                       const PointCloudRGBA::Ptr &cloud) {
  PCLVisualizer viewer = createViewer("Samples");

  // draw the point cloud
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(
      cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb,
                                           "registered point cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
      "registered point cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0,
      "registered point cloud");

  // draw the samples
  viewer->addPointCloud<pcl::PointXYZRGBA>(samples_cloud, "samples cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "samples cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 1.0, "samples cloud");

  runViewer(viewer);
}

void Plot::plotNormals(const Cloud &cloud_cam, bool draw_camera_cone) {
  const int num_clouds = cloud_cam.getViewPoints().cols();
  std::vector<PointCloudPointNormal::Ptr> clouds;
  clouds.resize(num_clouds);

  for (int i = 0; i < num_clouds; i++) {
    PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
    clouds[i] = cloud;
  }

  for (int i = 0; i < cloud_cam.getNormals().cols(); i++) {
    pcl::PointNormal p;
    p.x = cloud_cam.getCloudProcessed()->points[i].x;
    p.y = cloud_cam.getCloudProcessed()->points[i].y;
    p.z = cloud_cam.getCloudProcessed()->points[i].z;
    p.normal_x = cloud_cam.getNormals()(0, i);
    p.normal_y = cloud_cam.getNormals()(1, i);
    p.normal_z = cloud_cam.getNormals()(2, i);

    for (int j = 0; j < cloud_cam.getCameraSource().rows(); j++) {
      if (cloud_cam.getCameraSource()(j, i) == 1) {
        clouds[j]->push_back(p);
      }
    }
  }

  double colors[6][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
                         {1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}};
  double normal_colors[6][3] = {{0.5, 0.0, 0.0}, {0.0, 0.5, 0.0},
                                {0.0, 0.0, 0.5}, {0.5, 0.5, 0.0},
                                {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};

  if (num_clouds == 1) {
    normal_colors[0][0] = 0.0;
    normal_colors[0][2] = 1.0;
  }

  PCLVisualizer viewer = createViewer("Normals");
  viewer->setBackgroundColor(1.0, 1.0, 1.0);
  for (int i = 0; i < num_clouds; i++) {
    std::string cloud_name = "cloud_" + std::to_string(i);
    std::string normals_name = "normals_" + std::to_string(i);
    int color_id = i % 6;
    viewer->addPointCloud<pcl::PointNormal>(clouds[i], cloud_name);
    viewer->addPointCloudNormals<pcl::PointNormal>(clouds[i], 1, 0.01,
                                                   normals_name);
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, colors[color_id][0],
        colors[color_id][1], colors[color_id][2], cloud_name);
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, normal_colors[color_id][0],
        normal_colors[color_id][1], normal_colors[color_id][2], normals_name);

    // draw camera position as a cube
    if (draw_camera_cone) {
      const Eigen::Vector3d &cam_pos = cloud_cam.getViewPoints().col(i);
      Eigen::Vector4f centroid_4d;
      pcl::compute3DCentroid(*clouds[i], centroid_4d);
      Eigen::Vector3d centroid;
      centroid << centroid_4d(0), centroid_4d(1), centroid_4d(2);
      Eigen::Vector3d cone_dir = centroid - cam_pos;
      cone_dir.normalize();
      pcl::ModelCoefficients coeffs;
      coeffs.values.push_back(cam_pos(0));
      coeffs.values.push_back(cam_pos(1));
      coeffs.values.push_back(cam_pos(2));
      coeffs.values.push_back(cone_dir(0));
      coeffs.values.push_back(cone_dir(1));
      coeffs.values.push_back(cone_dir(2));
      coeffs.values.push_back(20.0);
      std::string cone_name = "cam" + std::to_string(i);
      viewer->addCone(coeffs, cone_name, 0);
      viewer->setShapeRenderingProperties(
          pcl::visualization::PCL_VISUALIZER_COLOR, normal_colors[color_id][0],
          normal_colors[color_id][1], normal_colors[color_id][2], cone_name);
    }
  }

  runViewer(viewer);
}

void Plot::plotNormals(const PointCloudRGBA::Ptr &cloud,
                       const PointCloudRGBA::Ptr &cloud_samples,
                       const Eigen::Matrix3Xd &normals) {
  PointCloudPointNormal::Ptr normals_cloud(new PointCloudPointNormal);
  for (int i = 0; i < normals.cols(); i++) {
    pcl::PointNormal p;
    p.x = cloud_samples->points[i].x;
    p.y = cloud_samples->points[i].y;
    p.z = cloud_samples->points[i].z;
    p.normal_x = normals(0, i);
    p.normal_y = normals(1, i);
    p.normal_z = normals(2, i);
    normals_cloud->points.push_back(p);
  }
  std::cout << "Drawing " << normals_cloud->size() << " normals\n";

  double red[3] = {1.0, 0.0, 0.0};
  double blue[3] = {0.0, 0.0, 1.0};

  PCLVisualizer viewer = createViewer("Normals");

  // draw the point cloud
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(
      cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb,
                                           "registered point cloud");

  // draw the normals
  addCloudNormalsToViewer(viewer, normals_cloud, 2, blue, red,
                          std::string("cloud"), std::string("normals"));

  runViewer(viewer);
}

void Plot::plotNormals(const PointCloudRGBA::Ptr &cloud,
                       const Eigen::Matrix3Xd &normals) {
  PointCloudPointNormal::Ptr normals_cloud(new PointCloudPointNormal);
  for (int i = 0; i < normals.cols(); i++) {
    pcl::PointNormal p;
    p.x = cloud->points[i].x;
    p.y = cloud->points[i].y;
    p.z = cloud->points[i].z;
    p.normal_x = normals(0, i);
    p.normal_y = normals(1, i);
    p.normal_z = normals(2, i);
    normals_cloud->points.push_back(p);
  }
  std::cout << "Drawing " << normals_cloud->size() << " normals\n";

  double red[3] = {1.0, 0.0, 0.0};
  double blue[3] = {0.0, 0.0, 1.0};

  PCLVisualizer viewer = createViewer("Normals");

  // draw the point cloud
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(
      cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb,
                                           "registered point cloud");

  // draw the normals
  addCloudNormalsToViewer(viewer, normals_cloud, 2, blue, red,
                          std::string("cloud"), std::string("normals"));

  runViewer(viewer);
}

void Plot::plotNormals(const Eigen::Matrix3Xd &pts,
                       const Eigen::Matrix3Xd &normals) {
  PointCloudPointNormal::Ptr normals_cloud(new PointCloudPointNormal);
  for (int i = 0; i < normals.cols(); i++) {
    pcl::PointNormal p;
    p.x = pts(0, i);
    p.y = pts(1, i);
    p.z = pts(2, i);
    p.normal_x = normals(0, i);
    p.normal_y = normals(1, i);
    p.normal_z = normals(2, i);
    normals_cloud->points.push_back(p);
  }
  std::cout << "Drawing " << normals_cloud->size() << " normals\n";

  double red[3] = {1.0, 0.0, 0.0};
  double blue[3] = {0.0, 0.0, 1.0};

  PCLVisualizer viewer = createViewer("Normals");
  addCloudNormalsToViewer(viewer, normals_cloud, 2, blue, red,
                          std::string("cloud"), std::string("normals"));
  runViewer(viewer);
}

void Plot::plotLocalAxes(const std::vector<candidate::LocalFrame> &frames,
                         const PointCloudRGBA::Ptr &cloud) {
  PCLVisualizer viewer = createViewer("Local Frames");
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, "registered point cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
      "registered point cloud");

  for (int i = 0; i < frames.size(); i++) {
    const candidate::LocalFrame &frame = frames[i];
    pcl::PointXYZ p, q, r;
    p.getVector3fMap() = frame.getSample().cast<float>();
    q.x = p.x + 0.02 * frame.getCurvatureAxis()(0);
    q.y = p.y + 0.02 * frame.getCurvatureAxis()(1);
    q.z = p.z + 0.02 * frame.getCurvatureAxis()(2);
    r.x = p.x + 0.02 * frame.getNormal()(0);
    r.y = p.y + 0.02 * frame.getNormal()(1);
    r.z = p.z + 0.02 * frame.getNormal()(2);
    const std::string str_id = std::to_string(i);
    viewer->addLine<pcl::PointXYZ>(p, q, 0, 0, 255, "curvature_axis_" + str_id);
    viewer->addLine<pcl::PointXYZ>(p, r, 255, 0, 0, "normal_axis_" + str_id);
  }

  runViewer(viewer);
}

void Plot::plotCameraSource(const Eigen::VectorXi &pts_cam_source_in,
                            const PointCloudRGBA::Ptr &cloud) {
  PointCloudRGBA::Ptr left_cloud(new PointCloudRGBA);
  PointCloudRGBA::Ptr right_cloud(new PointCloudRGBA);

  for (int i = 0; i < pts_cam_source_in.size(); i++) {
    if (pts_cam_source_in(i) == 0)
      left_cloud->points.push_back(cloud->points[i]);
    else if (pts_cam_source_in(i) == 1)
      right_cloud->points.push_back(cloud->points[i]);
  }

  PCLVisualizer viewer = createViewer("Camera Sources");
  viewer->addPointCloud<pcl::PointXYZRGBA>(left_cloud, "left point cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "left point cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0,
      "left point cloud");
  viewer->addPointCloud<pcl::PointXYZRGBA>(right_cloud, "right point cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "right point cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0,
      "right point cloud");
  runViewer(viewer);
}

void Plot::addCloudNormalsToViewer(PCLVisualizer &viewer,
                                   const PointCloudPointNormal::Ptr &cloud,
                                   double line_width, double *color_cloud,
                                   double *color_normals,
                                   const std::string &cloud_name,
                                   const std::string &normals_name) {
  viewer->addPointCloud<pcl::PointNormal>(cloud, cloud_name);
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, color_cloud[0], color_cloud[1],
      color_cloud[2], cloud_name);
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, cloud_name);
  viewer->addPointCloudNormals<pcl::PointNormal>(cloud, 1, 0.01, normals_name);
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, normals_name);
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, color_normals[0],
      color_normals[1], color_normals[2], normals_name);
}

void Plot::runViewer(PCLVisualizer &viewer) {
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
  }

  viewer->close();
}

PCLVisualizer Plot::createViewer(std::string title) {
  PCLVisualizer viewer(new pcl::visualization::PCLVisualizer(title));
  viewer->setPosition(0, 0);
  viewer->setSize(640, 480);
  viewer->setBackgroundColor(1.0, 1.0, 1.0);
  viewer->registerKeyboardCallback(&Plot::keyboardEventOccurred, *this,
                                   (void *)viewer.get());

  return viewer;
}

void Plot::keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                                 void *viewer_void) {
  pcl::visualization::PCLVisualizer *viewer =
      static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);
  if (event.getKeySym() == "a" && event.keyDown()) {
    if (viewer->contains("ref")) {
      viewer->removeCoordinateSystem("ref");
    } else {
      viewer->addCoordinateSystem(0.1, "ref");
    }
  }
}

void Plot::plotCloud(const PointCloudRGBA::Ptr &cloud_rgb,
                     const std::string &title) {
  PCLVisualizer viewer = createViewer(title);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(
      cloud_rgb);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud_rgb, rgb,
                                           "registered point cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
      "registered point cloud");
  runViewer(viewer);
}

pcl::PointXYZRGBA Plot::eigenVector3dToPointXYZRGBA(const Eigen::Vector3d &v) {
  pcl::PointXYZRGBA p;
  p.x = v(0);
  p.y = v(1);
  p.z = v(2);
  return p;
}

}  // namespace util
}  // namespace gpd
