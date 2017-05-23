#ifndef DATA_GENERATOR_H_
#define DATA_GENERATOR_H_


// Caffe
#include <caffe/util/io.hpp>
#include <caffe/util/db.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

// ROS
#include <ros/ros.h>

// Custom
#include <gpg/cloud_camera.h>
#include <gpg/candidates_generator.h>

#include "../gpd/learning.h"


struct Instance
{
  cv::Mat image_;
  bool label_;

  Instance(const cv::Mat& image, bool label) : image_(image), label_(label) { }
};


class DataGenerator
{
  public:

    /**
     * \brief Constructor.
     * \param node ROS node handle
     */
    DataGenerator(ros::NodeHandle& node);

    /**
     * \brief Destructor.
     */
    ~DataGenerator();

    /**
     * \brief Create training data.
     * \param node ROS node handle
     */
    void createTrainingData(ros::NodeHandle& node);


  private:

    /**
     * \brief Create a grasp candidates generator.
     * \param node ROS node handle
     * \return the grasp candidates generator
     */
    CandidatesGenerator* createCandidatesGenerator(ros::NodeHandle& node);

    /**
     * \brief Create a learning object to generate grasp images.
     * \param node ROS node handle
     * \return the learning object
     */
    Learning* createLearning(ros::NodeHandle& node);

    /**
     * \brief Create grasp images.
     * \param node ROS node handle
     * \return the grasp candidates generator
     */
    bool createGraspImages(CloudCamera& cloud_cam, std::vector<Grasp>& grasps_out, std::vector<cv::Mat>& images_out);

    /**
     * \brief Load a point cloud given ROS launch parameters.
     * \param node ROS node handle
     * \return the point cloud
     */
    CloudCamera loadCloudCameraFromFile(ros::NodeHandle& node);

    /**
     * \brief Load a point cloud and surface normals given ROS launch parameters.
     * \param mesh_file_path location of the point cloud file
     * \param normals_file_path location of the surface normals file
     * \return the point cloud with surface normals
     */
    CloudCamera loadMesh(const std::string& mesh_file_path, const std::string& normals_file_path);

    /**
     * \brief Load all point cloud files from a folder.
     * \param cloud_folder location of the folder
     * \return list of point cloud files
     */
    std::vector<boost::filesystem::path> loadPointCloudFiles(const std::string& cloud_folder);

    /**
     * \brief Load object names from a file.
     * \param objects_file_location location of the file
     * \return list of object names
     */
    std::vector<std::string> loadObjectNames(const std::string& objects_file_location);

    /**
     * \brief Balance the number of positive and negative examples.
     * \param max_grasps_per_view maximum number of examples per camera view
     * \param positives_in positive examples
     * \param negatives_in negative examples
     * \param[out] positives_out positive examples after balancing
     * \param[out] negatives_out negative examples after balancing
     */
    void balanceInstances(int max_grasps_per_view, const std::vector<int>& positives_in,
      const std::vector<int>& negatives_in, std::vector<int>& positives_out, std::vector<int>& negatives_out);

    /**
     * \brief Add examples to the dataset.
     * \param grasps grasp candidates
     * \param images grasp images
     * \param positives indices of positive/viable grasps
     * \param negatives indices of negative/non-viable grasps
     * \param dataset the dataset that the examples are added to
     */
    void addInstances(const std::vector<Grasp>& grasps, const std::vector<cv::Mat>& images,
      const std::vector<int>& positives, const std::vector<int>& negatives, std::vector<Instance>& dataset);

    /**
     * \brief Store the dataset as an LMDB.
     * \param dataset the dataset
     * \param file_location location where the LMDB is stored
     */
    void storeLMDB(const std::vector<Instance>& dataset, const std::string& file_location);

    CandidatesGenerator* candidates_generator_; ///< object to generate grasp candidates
    Learning* learning_; ///< object to generate grasp images
};


#endif /* GENERATOR_H_ */
