/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016, Andreas ten Pas
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


#ifndef LEARNING_H_
#define LEARNING_H_

#include <fstream>
#include <iostream>
#include <set>
#include <sys/stat.h>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/unordered_set.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>

#include <gpg/cloud_camera.h>
#include <gpg/eigen_utils.h>
#include <gpg/grasp.h>
#include <gpg/grasp_set.h>

#include <gpd/grasp_image_15_channels.h>


typedef std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd> Matrix3XdPair;


/** Learning class
 *
 * \brief Create images for classification.
 * 
 * This class is used to create images for the input layer of a convolutional neural network. Each image represents a
 * grasp candidate. We call these "grasp images".
 * 
 */
class Learning
{
  public:

    /**
     * \brief Parameters for the grasp images.
     */
    struct ImageParameters
    {
      double outer_diameter_; ///< the maximum robot hand aperture
      double depth_; ///< the hand depth (length of fingers)
      double height_; ///< the hand extends plus/minus this value along the hand axis
      int size_; ///< the size of the image in pixels (for one side)
      int num_channels_; ///< the number of channels in the grasp image
    };

    /**
     * \brief Constructor.
     * \param params parameters for grasp images
     * \param num_threads number of CPU threads to be used
     * \param is_plotting if images are visualized
     * \param remove_plane if the (table) plane is removed before calculating images
     */
    Learning(const ImageParameters& params, int num_threads, int num_orientations, bool is_plotting, bool remove_plane)
      : image_params_(params), num_threads_(num_threads), num_orientations_(num_orientations),
        is_plotting_(is_plotting), remove_plane_(remove_plane) { }

    /**
     * \brief Match grasps with their corresponding grasp images.
     * \param hand_set_list list of grasp candidate sets
     * \param images list of grasp images
     * \param[out] grasps_out list of grasps matched to images
     * \param[out] images_out list of images matched to grasps
     */
    void extractGraspsAndImages(const std::vector<GraspSet>& hand_set_list, const std::vector<cv::Mat>& images,
      std::vector<Grasp>& grasps_out, std::vector<cv::Mat>& images_out);

    /**
     * \brief Create a list of grasp images for a given list of grasp candidates.
     * \param cloud_cam the point cloud
     * \param hand_set_list the list of grasp candidates
     * \return the list of grasp images
     */
    std::vector<cv::Mat> createImages(const CloudCamera& cloud_cam, const std::vector<GraspSet>& hand_set_list) const;


  private:

    /**
     * \brief Create grasp images with one or three channels.
     * \param hand_set_list the list of grasp candidate sets
     * \param nn_points_list the list of point neighborhoods
     * \param is_valid the list of booleans indicating for each neighborhood if it is valid or not
     * \param image_dims the image dimensions
     * \return the grasp images
     */
    std::vector<cv::Mat> createImages1or3Channels(const std::vector<GraspSet>& hand_set_list,
      const std::vector<PointList>& nn_points_list, const bool* is_valid, const Eigen::Vector3d& image_dims) const;

    /**
     * \brief Create a one or three channels grasp image.
     * \param point_list the point neighborhood
     * \param hand the grasp
     * \return the grasp image
     */
    cv::Mat createImage1or3Channels(const PointList& point_list, const Grasp& hand) const;

    /**
     * \brief Create grasp images with 15 channels.
     * \param hand_set_list the list of grasp candidate sets
     * \param nn_points_list the list of point neighborhoods
     * \param is_valid the list of booleans indicating for each neighborhood if it is valid or not
     * \param image_dims the image dimensions
     * \return the grasp images
     */
    std::vector<cv::Mat> createImages15Channels(const std::vector<GraspSet>& hand_set_list,
      const std::vector<PointList>& nn_points_list, const bool* is_valid, const Eigen::Vector3d& image_dims) const;

    /**
     * \brief Create a 15 channels grasp image.
     * \param point_list the point neighborhood
     * \param shadow the shadow of the point neighborhood
     * \param hand the grasp
     * \return the grasp image
     */
    cv::Mat createImage15Channels(const PointList& point_list, const Eigen::Matrix3Xd& shadow, const Grasp& hand)
      const;

    /**
     * \brief Transform a given list of points to the unit image.
     * \param point_list the list of points
     * \param hand the grasp
     * \return the transformed points and their surface normals
     */
    Matrix3XdPair transformToUnitImage(const PointList& point_list, const Grasp& hand) const;

    /**
     * \brief Find points that lie in the closing region of the robot hand.
     * \param hand the grasp
     * \param points the points to be checked
     * \return the indices of the points that lie inside the closing region
     */
    std::vector<int> findPointsInUnitImage(const Grasp& hand, const Eigen::Matrix3Xd& points) const;

    /**
     * \brief Transform points to the unit image.
     * \param hand the grasp
     * \param points the points
     * \param indices the indices of the points to be transformed
     * \return the transformed points
     */
    Eigen::Matrix3Xd transformPointsToUnitImage(const Grasp& hand, const Eigen::Matrix3Xd& points,
      const std::vector<int>& indices) const;

    /**
     * \brief Transform a list of points into a frame.
     * \param point_list the points to be transformed
     * \param centroid the origin of the frame
     * \param rotation the orientation of the frame
     * \return the transformed points and their surface normals
     */
    Matrix3XdPair transformToHandFrame(const PointList& point_list, const Eigen::Vector3d& centroid,
      const Eigen::Matrix3d& rotation) const;

    int num_threads_; ///< number of threads
    int num_orientations_; ///< number of hand orientations
    ImageParameters image_params_; ///< parameters of the grasp image
    bool is_plotting_; ///< if grasp images are visualized
    bool remove_plane_; ///< if the largest plane is removed from the point cloud
};

#endif /* LEARNING_H_ */
