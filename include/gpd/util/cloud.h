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

#ifndef CLOUD_H_
#define CLOUD_H_

#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/normal_refinement.h>
#include <pcl/filters/random_sample.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/impl/kdtree.hpp>

#if defined(USE_PCL_GPU)  // switch include ordering if compilation fails
#include <pcl/gpu/containers/device_array.hpp>
#include <pcl/gpu/features/features.hpp>
#endif

#include <gpd/util/eigen_utils.h>

namespace gpd {
namespace util {

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGB;
typedef pcl::PointCloud<pcl::PointNormal> PointCloudPointNormal;
typedef pcl::PointCloud<pcl::Normal> PointCloudNormal;

/**
 *
 * \brief Multi-view point cloud
 *
 * Stores and processes a point cloud that has been observed from one or
 * multiple camera view points. The raw point cloud is stored in
 * `cloud_original_` and the processed point cloud in `cloud_processed_`.
 *
 */
class Cloud {
 public:
  /**
   * \brief Comparator for checking uniqueness of two 3D-vectors.
   */
  struct UniqueVectorComparator {
    /**
     * \brief Compares two 3D-vectors for uniqueness.
     * \param a the first 3D-vector
     * \param b the second 3D-vector
     * \return true if they differ in at least one element, false if all
     * elements are equal
     */
    bool operator()(const Eigen::Vector3i &a, const Eigen::Vector3i &b) {
      for (int i = 0; i < a.size(); i++) {
        if (a(i) != b(i)) {
          return a(i) < b(i);
        }
      }

      return false;
    }
  };

  /**
   * \brief Comparator for checking uniqueness of two 4D-vectors.
   */
  struct UniqueVector4First3Comparator {
    /**
     * \brief Compares two 4D-vectors for uniqueness (ignores the last element).
     * \param a the first 4D-vector
     * \param b the second 4D-vector
     * \return true if they differ in at least one of the first three elements,
     * false otherwise
     */
    bool operator()(const Eigen::Vector4i &a, const Eigen::Vector4i &b) {
      for (int i = 0; i < a.size() - 1; i++) {
        if (a(i) != b(i)) {
          return true;
        }
      }

      return false;
    }
  };

  /**
   * \brief Constructor.
   */
  Cloud();

  /**
   * \brief Constructor.
   * \param cloud the point cloud (of size n)
   * \param camera_source the camera source for each point in the cloud (size: k
   * x n)
   * \param view_points the origins of the cameras (size: 3 x k)
   */
  Cloud(const PointCloudRGB::Ptr &cloud, const Eigen::MatrixXi &camera_source,
        const Eigen::Matrix3Xd &view_points);

  /**
   * \brief Constructor.
   * \param cloud the point cloud with surface normals (of size n)
   * \param camera_source the camera source for each point in the cloud (size: k
   * x n)
   * \param view_points the origins of the cameras (size: 3 x k)
   */
  Cloud(const PointCloudPointNormal::Ptr &cloud,
        const Eigen::MatrixXi &camera_source,
        const Eigen::Matrix3Xd &view_points);

  /**
   * \brief Constructor for a two camera setup (left and right camera).
   * \param cloud the point cloud (of size n)
   * \param size_left_cloud the size of the point cloud from the left camera
   * \param view_points the origins of the cameras (size: 3 x k)
   */
  Cloud(const PointCloudRGB::Ptr &cloud, int size_left_cloud,
        const Eigen::Matrix3Xd &view_points);

  /**
   * \brief Constructor for a two cameras setup (left and right camera).
   * \param cloud the point cloud with surface normals (of size n)
   * \param size_left_cloud the size of the point cloud from the left camera
   * \param view_points the origins of the cameras (size: 3 x k)
   */
  Cloud(const PointCloudPointNormal::Ptr &cloud, int size_left_cloud,
        const Eigen::Matrix3Xd &view_points);

  /**
   * \brief Constructor for point cloud files (*.pcd).
   * \param filename the location of the point cloud file
   * \param view_points the origins of the cameras (size: 3 x k)
   */
  Cloud(const std::string &filename, const Eigen::Matrix3Xd &view_points);

  /**
   * \brief Constructor for point cloud files (*.pcd) from a two cameras setup.
   * \param filename_left the location of the point cloud file from the first
   * camera
   * \param filename_right the location of the point cloud file from the second
   * camera
   * \param view_points the origins of the cameras (size: 3 x k)
   */
  Cloud(const std::string &filename_left, const std::string &filename_right,
        const Eigen::Matrix3Xd &view_points);

  /**
   * \brief Remove NANs from the point cloud.
   */
  void removeNans();

  /**
   * \brief Remove statistical outliers from the point cloud.
   */
  void removeStatisticalOutliers();

  void refineNormals(int k = 10);

  /**
   * \brief Filter out points in the point cloud that lie outside the workspace
   * dimensions.
   * \param[in] workspace a 6-D vector containing the workspace limits: [minX,
   * maxX, minY, maxY, minZ, maxZ]
   */
  void filterWorkspace(const std::vector<double> &workspace);

  /**
   * \brief Filter out samples that lie outside the workspace dimensions.
   * \param[in] workspace a 6-D vector containing the workspace limits: [minX,
   * maxX, minY, maxY, minZ, maxZ]
   */
  void filterSamples(const std::vector<double> &workspace);

  /**
   * \brief Voxelize the point cloud and keep track of the camera source for
   * each voxel.
   * \param[in] cell_size the size of each voxel
   */
  void voxelizeCloud(float cell_size);

  /**
   * \brief Subsample the point cloud according to the uniform distribution.
   * \param[in] num_samples the number of samples to draw from the point cloud
   */
  void subsampleUniformly(int num_samples);

  /**
   * \brief Subsample the samples according to the uniform distribution.
   * \param[in] num_samples the number of samples to draw from the samples
   */
  void subsampleSamples(int num_samples);

  /**
   * \brief Subsample the sample indices according to the uniform distribution.
   * \param[in] num_samples the number of samples to draw from the indices
   */
  void subsampleSampleIndices(int num_samples);

  /**
   * \brief Sample above the support plane.
   */
  void sampleAbovePlane();

  /**
   * \brief Subsample the cloud.
   *
   * A helper function that subsamples the cloud, the samples, or the sample
   * indices depending on which component is available.
   * \param[in] num_samples the number of samples to draw
   */
  void subsample(int num_samples);

  /**
   * \brief Estimate surface normals for the point cloud.
   * Switches the normals estimation method based on if the point cloud is
   * organized or not. \param[in] num_threads the number of CPU threads to be
   * used in the calculation
   */
  void calculateNormals(int num_threads, double radius);

  /**
   * \brief Calculate surface normals for an organized point cloud.
   */
  void calculateNormalsOrganized();

  /**
   * \brief Calculate surface normals for an unorganized point cloud.
   * \param[in] num_threads the number of CPU threads to be used in the
   * calculation
   */
  void calculateNormalsOMP(int num_threads, double radius);

#if defined(USE_PCL_GPU)
  void calculateNormalsGPU();
#endif

  /**
   * \brief Reverse direction of surface normals (if a normal does not point to
   * at least one camera).
   */
  void reverseNormals();

  /**
   * \brief Set surface normals from a file.
   * \param[in] filename the location of the file
   */
  void setNormalsFromFile(const std::string &filename);

  /**
   * \brief Write surface normals to a file.
   * \param filename the location of the file
   * \param normals the surface normals
   */
  void writeNormalsToFile(const std::string &filename,
                          const Eigen::Matrix3Xd &normals);

  /**
   * \brief Return the camera source matrix.
   * \return the camera source matrix (k x n)
   */
  const Eigen::MatrixXi &getCameraSource() const { return camera_source_; }

  /**
   * \brief Return the preprocessed point cloud.
   * \return the point cloud
   */
  const PointCloudRGB::Ptr &getCloudProcessed() const {
    return cloud_processed_;
  }

  /**
   * \brief Return the original point cloud.
   * \return the point cloud
   */
  const PointCloudRGB::Ptr &getCloudOriginal() const { return cloud_original_; }

  /**
   * \brief Return the original point cloud.
   * \return the point cloud
   */
  const std::vector<int> &getSampleIndices() const { return sample_indices_; }

  /**
   * \brief Return the surface normals.
   * \return the surface normals (size: 3 x n)
   */
  const Eigen::Matrix3Xd &getNormals() const { return normals_; }

  /**
   * \brief Return the samples.
   * \return the samples (size: 3 x n)
   */
  const Eigen::Matrix3Xd &getSamples() const { return samples_; }

  /**
   * \brief Return the sample indices.
   * \return the sample indices (size: n)
   */
  void setSampleIndices(const std::vector<int> &sampleIndices) {
    sample_indices_ = sampleIndices;
  }

  /**
   * \brief Set the samples.
   * \return the samples (size: 3 x n)
   */
  void setSamples(const Eigen::Matrix3Xd &samples);

  /**
   * \brief Set the surface normals.
   * \return the surface normals (size: 3 x n)
   */
  void setNormals(const Eigen::Matrix3Xd &normals) { normals_ = normals; }

  /**
   * \brief Get the view points of the cameras.
   * \return the origins of the cameras (size: 3 x k)
   */
  const Eigen::Matrix3Xd &getViewPoints() const { return view_points_; }

  /**
   * \brief Set the camera view points.
   * \return the origins of the cameras (size: 3 x k)
   */
  void setViewPoints(const Eigen::Matrix3Xd &view_points) {
    view_points_ = view_points;
  }

  /**
   * \brief Load a point cloud from a file.
   * \param filename the location of the file
   * \return the point cloud
   */
  PointCloudRGB::Ptr loadPointCloudFromFile(const std::string &filename) const;

 private:
  std::vector<std::vector<int>> convertCameraSourceMatrixToLists();

  PointCloudRGB::Ptr cloud_processed_;
  PointCloudRGB::Ptr cloud_original_;

  // binary matrix: (i,j) = 1 if point j is seen by camera i, 0 otherwise
  Eigen::MatrixXi camera_source_;
  Eigen::Matrix3Xd normals_;
  Eigen::Matrix3Xd view_points_;

  std::vector<int> sample_indices_;
  Eigen::Matrix3Xd samples_;
};

}  // namespace util
}  // namespace gpd

#endif /* CLOUD_H_ */
