/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2017, Andreas ten Pas
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

#ifndef DATA_GENERATOR_H_
#define DATA_GENERATOR_H_

#include <memory>
#include <vector>

#include <Eigen/Dense>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/hdf.hpp>
#include <opencv2/opencv.hpp>

// PCL
#include <pcl/common/transforms.h>

// Grasp Pose Generator
#include <gpd/util/cloud.h>
#include <gpd/util/config_file.h>

// Custom
#include <gpd/grasp_detector.h>

namespace gpd {

typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGB;

struct Instance {
  std::unique_ptr<cv::Mat> image_;
  bool label_;

  Instance(std::unique_ptr<cv::Mat> image, bool label)
      : image_(std::move(image)), label_(label) {}
};

class DataGenerator {
 public:
  /**
   * \brief Constructor.
   * \param node ROS node handle
   */
  DataGenerator(const std::string &config_filename);

  void generateDataBigbird();

  /**
   * \brief Create training data.
   * \param node ROS node handle
   */
  void generateData();

  util::Cloud createMultiViewCloud(const std::string &object, int camera,
                                   const std::vector<int> angles,
                                   int reference_camera) const;

 private:
  void createDatasetsHDF5(const std::string &filepath, int num_data);

  void reshapeHDF5(const std::string &in, const std::string &out,
                   const std::string &dataset, int num, int chunk_size,
                   int max_in_memory);

  /**
   * \brief Load a point cloud and surface normals given ROS launch parameters.
   * \param mesh_file_path location of the point cloud file
   * \param normals_file_path location of the surface normals file
   * \return the point cloud with surface normals
   */
  util::Cloud loadMesh(const std::string &mesh_file_path,
                       const std::string &normals_file_path);

  /**
   * \brief Load object names from a file.
   * \param objects_file_location location of the file
   * \return list of object names
   */
  std::vector<std::string> loadObjectNames(
      const std::string &objects_file_location);

  void splitInstances(const std::vector<int> &labels,
                      std::vector<int> &positives, std::vector<int> &negatives);

  /**
   * \brief Balance the number of positive and negative examples.
   * \param max_grasps_per_view maximum number of examples per camera view
   * \param positives_in positive examples
   * \param negatives_in negative examples
   * \param[out] positives_out positive examples after balancing
   * \param[out] negatives_out negative examples after balancing
   */
  void balanceInstances(int max_grasps_per_view,
                        const std::vector<int> &positives_in,
                        const std::vector<int> &negatives_in,
                        std::vector<int> &positives_out,
                        std::vector<int> &negatives_out);

  /**
   * \brief Add examples to the dataset.
   * \param grasps grasp candidates
   * \param images grasp images
   * \param positives indices of positive/viable grasps
   * \param negatives indices of negative/non-viable grasps
   * \param dataset the dataset that the examples are added to
   */
  void addInstances(const std::vector<std::unique_ptr<candidate::Hand>> &grasps,
                    std::vector<std::unique_ptr<cv::Mat>> &images,
                    const std::vector<int> &positives,
                    const std::vector<int> &negatives,
                    std::vector<Instance> &dataset);

  int insertIntoHDF5(const std::string &file_path,
                     const std::vector<Instance> &dataset, int offset);

  /**
   * \brief Store the dataset as an HDF5 file.
   * \param dataset the dataset
   * \param file_location location where the LMDB is stored
   */
  void storeHDF5(const std::vector<Instance> &dataset,
                 const std::string &file_location);

  void printMatrix(const cv::Mat &mat);

  void printMatrix15(const cv::Mat &mat);

  void copyMatrix(const cv::Mat &src, cv::Mat &dst, int idx_in, int *dims_img);

  Eigen::Matrix4f calculateTransform(const std::string &object, int camera,
                                     int angle, int reference_camera) const;

  Eigen::Matrix4f readPoseFromHDF5(const std::string &hdf5_filename,
                                   const std::string &dsname) const;

  std::unique_ptr<GraspDetector>
      detector_;  ///< object to generate grasp candidates and images

  std::string data_root_;
  std::string objects_file_location_;
  std::string output_root_;
  int num_views_per_object_;
  int min_grasps_per_view_;
  int max_grasps_per_view_;
  int chunk_size_;
  int max_in_memory_;
  int num_threads_;
  int num_samples_;
  double voxel_size_views_;
  double normals_radius_;
  bool remove_nans_;
  bool reverse_mesh_normals_;
  bool reverse_view_normals_;
  std::vector<int> test_views_;
  std::vector<int> all_cam_sources_;

  static const std::string IMAGES_DS_NAME;
  static const std::string LABELS_DS_NAME;
};

}  // namespace gpd

#endif /* GENERATOR_H_ */
