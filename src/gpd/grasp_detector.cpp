#include <gpd/grasp_detector.h>


GraspDetector::GraspDetector(ros::NodeHandle& node)
{
  Eigen::initParallel();

  // Create objects to store parameters.
  CandidatesGenerator::Parameters generator_params;
  HandSearch::Parameters hand_search_params;

  // Read hand geometry parameters.
  node.param("finger_width", hand_search_params.finger_width_, 0.01);
  node.param("hand_outer_diameter", hand_search_params.hand_outer_diameter_, 0.09);
  node.param("hand_depth", hand_search_params.hand_depth_, 0.06);
  node.param("hand_height", hand_search_params.hand_height_, 0.02);
  node.param("init_bite", hand_search_params.init_bite_, 0.015);
  outer_diameter_ = hand_search_params.hand_outer_diameter_;

  // Read local hand search parameters.
  node.param("nn_radius", hand_search_params.nn_radius_frames_, 0.01);
  node.param("num_orientations", hand_search_params.num_orientations_, 8);
  node.param("num_samples", hand_search_params.num_samples_, 500);
  node.param("num_threads", hand_search_params.num_threads_, 1);
  node.param("rotation_axis", hand_search_params.rotation_axis_, 2); // cannot be changed

  // Read plotting parameters.
  node.param("plot_samples", plot_samples_, false);
  node.param("plot_normals", plot_normals_, false);
  generator_params.plot_normals_ = plot_normals_;
  node.param("plot_filtered_grasps", plot_filtered_grasps_, false);
  node.param("plot_valid_grasps", plot_valid_grasps_, false);
  node.param("plot_clusters", plot_clusters_, false);
  node.param("plot_selected_grasps", plot_selected_grasps_, false);

  // Read general parameters.
  generator_params.num_samples_ = hand_search_params.num_samples_;
  generator_params.num_threads_ = hand_search_params.num_threads_;
  node.param("plot_candidates", generator_params.plot_grasps_, false);

  // Read preprocessing parameters.
  node.param("remove_outliers", generator_params.remove_statistical_outliers_, true);
  node.param("voxelize", generator_params.voxelize_, true);
  node.getParam("workspace", generator_params.workspace_);
  node.getParam("workspace_grasps", workspace_);

  // Create object to generate grasp candidates.
  candidates_generator_ = new CandidatesGenerator(generator_params, hand_search_params);

  // Read classification parameters and create classifier.
  std::string lenet_params_dir;
  node.param("lenet_params_dir", lenet_params_dir, std::string(""));
  node.param("min_score_diff", min_score_diff_, 500.0);
  node.param("create_image_batches", create_image_batches_, true);
  classifier_ = new Lenet(generator_params.num_threads_, lenet_params_dir);

  // Read grasp image parameters.
  node.param("image_outer_diameter", image_params_.outer_diameter_, hand_search_params.hand_outer_diameter_);
  node.param("image_depth", image_params_.depth_, hand_search_params.hand_depth_);
  node.param("image_height", image_params_.height_, hand_search_params.hand_height_);
  node.param("image_size", image_params_.size_, 60);
  node.param("image_num_channels", image_params_.num_channels_, 15);

  // Read learning parameters.
  bool remove_plane;
  node.param("remove_plane_before_image_calculation", remove_plane, false);

  // Create object to create grasp images from grasp candidates (used for classification)
  learning_ = new Learning(image_params_, hand_search_params.num_threads_, hand_search_params.num_orientations_, false,
                           remove_plane);

  // Read grasp filtering parameters
  node.param("filter_grasps", filter_grasps_, false);
  node.param("filter_half_antipodal", filter_half_antipodal_, false);
  std::vector<double> gripper_width_range(2);
  gripper_width_range[0] = 0.03;
  gripper_width_range[1] = 0.07;
  node.getParam("gripper_width_range", gripper_width_range);
  min_aperture_ = gripper_width_range[0];
  max_aperture_ = gripper_width_range[1];

  node.param("filter_table_side_grasps", filter_table_side_grasps_, false);
  node.getParam("vertical_axis", vert_axis_);
  node.param("angle_thresh", angle_thresh_, 0.1);
  node.param("table_height", table_height_, 0.5);
  node.param("table_thresh", table_thresh_, 0.05);

  // Read clustering parameters
  int min_inliers;
  node.param("min_inliers", min_inliers, 0);
  clustering_ = new Clustering(min_inliers);
  cluster_grasps_ = min_inliers > 0 ? true : false;

  // Read grasp selection parameters
  node.param("num_selected", num_selected_, 100);
}


std::vector<Grasp> GraspDetector::detectGrasps(const CloudCamera& cloud_cam)
{
  std::vector<Grasp> selected_grasps(0);

  // Check if the point cloud is empty.
  if (cloud_cam.getCloudOriginal()->size() == 0)
  {
    ROS_INFO("Point cloud is empty!");
    return selected_grasps;
  }

  Plot plotter;

  // Plot samples/indices.
  if (plot_samples_)
  {
    if (cloud_cam.getSamples().cols() > 0)
    {
      plotter.plotSamples(cloud_cam.getSamples(), cloud_cam.getCloudProcessed());
    }
    else if (cloud_cam.getSampleIndices().size() > 0)
    {
      plotter.plotSamples(cloud_cam.getSampleIndices(), cloud_cam.getCloudProcessed());
    }
  }

  if (plot_normals_)
  {
    std::cout << "Plotting normals for different camera sources\n";
    plotter.plotNormals(cloud_cam);
  }

  // 1. Generate grasp candidates.
  std::vector<GraspSet> candidates = generateGraspCandidates(cloud_cam);
  ROS_INFO_STREAM("Generated " << candidates.size() << " grasp candidate sets.");
  if (candidates.size() == 0)
  {
    return selected_grasps;
  }

  // 2.1 Prune grasp candidates based on min. and max. robot hand aperture and fingers below table surface.
  if (filter_grasps_)
  {
    candidates = filterGraspsWorkspace(candidates, workspace_);

    if (plot_filtered_grasps_)
    {
      const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
      plotter.plotFingers3D(candidates, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
        params.finger_width_, params.hand_depth_, params.hand_height_);
    }
  }

  // 2.2 Filter side grasps that are very close to the table.
  if (filter_table_side_grasps_)
  {
    candidates = filterSideGraspsCloseToTable(candidates);
  }

  // 2.3 Filter half grasps.
  if (filter_half_antipodal_)
  {
    candidates = filterHalfAntipodal(candidates);

    if (plot_filtered_grasps_)
    {
      const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
      plotter.plotFingers3D(candidates, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
        params.finger_width_, params.hand_depth_, params.hand_height_);
    }
  }

  // 3. Classify each grasp candidate. (Note: switch from a list of hypothesis sets to a list of grasp hypotheses)
  std::vector<Grasp> valid_grasps = classifyGraspCandidates(cloud_cam, candidates);
  ROS_INFO_STREAM("Selected " << valid_grasps.size() << " valid grasps after predicting their scores.");

  if (valid_grasps.size() <= 2)
  {
    std::cout << "Not enough valid grasps predicted! Using all grasps from previous step.\n";
    valid_grasps = extractHypotheses(candidates);
  }

  // 4. Cluster the grasps.
  std::vector<Grasp> clustered_grasps;

  if (cluster_grasps_)
  {
    clustered_grasps = findClusters(valid_grasps);
    ROS_INFO_STREAM("Found " << clustered_grasps.size() << " clusters.");
    if (clustered_grasps.size() <= 3)
    {
      std::cout << "Not enough clusters found! Using all grasps from previous step.\n";
      clustered_grasps = valid_grasps;
    }

    if (plot_clusters_)
    {
      const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
      plotter.plotFingers3D(clustered_grasps, cloud_cam.getCloudOriginal(), "Clustered Grasps",
        params.hand_outer_diameter_, params.finger_width_, params.hand_depth_, params.hand_height_);
    }
  }
  else
  {
    clustered_grasps = valid_grasps;
  }

//  // 5. Select highest-scoring grasps.
//  if (clustered_grasps.size() > num_selected_)
//  {
//    std::cout << "Partial Sorting the grasps based on their score ... \n";
//    std::partial_sort(clustered_grasps.begin(), clustered_grasps.begin() + num_selected_, clustered_grasps.end(),
//      isScoreGreater);
//    selected_grasps.assign(clustered_grasps.begin(), clustered_grasps.begin() + num_selected_);
//  }
//  else
//  {
//    std::cout << "Sorting the grasps based on their score ... \n";
//    std::sort(clustered_grasps.begin(), clustered_grasps.end(), isScoreGreater);
//    selected_grasps = clustered_grasps;
//  }

  std::cout << "==== Selected grasps ====\n";
  selected_grasps = clustered_grasps;
  std::sort(selected_grasps.begin(), selected_grasps.end(), isScoreGreater);
  for (int i = 0; i < selected_grasps.size(); i++)
  {
    std::cout << "Grasp " << i << ": " << selected_grasps[i].getScore() << "\n";
  }

  ROS_INFO_STREAM("Selected the " << selected_grasps.size() << " highest scoring grasps.");

  if (plot_selected_grasps_)
  {
    const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
    plotter.plotFingers3D(selected_grasps, cloud_cam.getCloudOriginal(), "Selected Grasps", params.hand_outer_diameter_,
      params.finger_width_, params.hand_depth_, params.hand_height_);
  }

  return selected_grasps;
}


std::vector<GraspSet> GraspDetector::generateGraspCandidates(const CloudCamera& cloud_cam)
{
  return candidates_generator_->generateGraspCandidateSets(cloud_cam);
}


void GraspDetector::preprocessPointCloud(CloudCamera& cloud_cam)
{
  candidates_generator_->preprocessPointCloud(cloud_cam);
}


std::vector<Grasp> GraspDetector::classifyGraspCandidates(const CloudCamera& cloud_cam,
  std::vector<GraspSet>& candidates)
{
  // Create a grasp image for each grasp candidate.
  double t0 = omp_get_wtime();
  std::cout << "Creating grasp images for classifier input ...\n";
  std::vector<float> scores;
  std::vector<Grasp> valid_grasps;
  int num_orientations = candidates[0].getHypotheses().size();

  // Create images in batches if required (less memory usage).
  if (create_image_batches_)
  {
      // TODO: implement this
//    int batch_size = classifier_->getBatchSize();
//    int num_iterations = (int) ceil(candidates.size() * num_orientations / (double) batch_size);
//    int step_size = (int) floor(batch_size / (double) num_orientations);
//    std::cout << " num_iterations: " << num_iterations << ", step_size: " << step_size << "\n";
//
//    // Process the grasp candidates in batches.
//    for (int i = 0; i < num_iterations; i++)
//    {
//      std::cout << i << "\n";
//      std::vector<GraspSet>::iterator start = candidates.begin() + i * step_size;
//      std::vector<GraspSet>::iterator stop;
//      if (i < num_iterations - 1)
//      {
//        stop = candidates.begin() + i * step_size + step_size;
//      }
//      else
//      {
//        stop = candidates.end();
//      }
//
//      std::vector<GraspSet> hand_set_sublist(start, stop);
//      std::vector<cv::Mat> image_list = learning_->createImages(cloud_cam, hand_set_sublist);
//
//      std::vector<Grasp> valid_grasps;
//      std::vector<cv::Mat> valid_images;
//      extractGraspsAndImages(candidates, image_list, valid_grasps, valid_images);
//
//      std::vector<float> scores_sublist = classifier_->classifyImages(valid_images);
//      scores.insert(scores.end(), scores_sublist.begin(), scores_sublist.end());
//      grasp_list.insert(grasp_list.end(), valid_grasps.begin(), valid_grasps.end());
//    }
  }
  else
  {
    // Create the grasp images.
    std::vector<cv::Mat> image_list = learning_->createImages(cloud_cam, candidates);
    std::cout << " Image creation time: " << omp_get_wtime() - t0 << std::endl;

    std::vector<cv::Mat> valid_images;
    extractGraspsAndImages(candidates, image_list, valid_grasps, valid_images);
    std::cout << " image_list: " << image_list.size() << ", valid_images: " << valid_images.size()
      << ", valid_grasps: " << valid_grasps.size() << std::endl;

    // Classify the grasp images.
    double t0_prediction = omp_get_wtime();
    scores = classifier_->classifyImages(valid_images);
    std::cout << " Prediction time: " << omp_get_wtime() - t0 << std::endl;

    for (int i = 0; i < valid_grasps.size(); i++)
    {
      valid_grasps[i].setScore(scores[i]);
    }
  }

  // Select the <num_selected_>-highest scoring grasps.
  std::cout << "Selecting the " << num_selected_ << " highest scoring grasps ..." << std::endl;
  int middle = std::min((int) valid_grasps.size(), num_selected_);
  std::partial_sort(valid_grasps.begin(), valid_grasps.begin() + middle, valid_grasps.end(), isScoreGreater);
  std::vector<Grasp> selected_grasps(valid_grasps.begin(), valid_grasps.begin() + middle);

  for (int i = 0; i < middle; i++)
  {
    std::cout << " grasp #" << i << ", score: " << valid_grasps[i].getScore() << ", " << selected_grasps[i].getScore() << "\n";
  }

  // Select grasps with a score of at least <min_score_diff_>.
//  std::vector<Grasp> valid_grasps;
//
//  for (int i = 0; i < grasp_list.size(); i++)
//  {
//    std::cout << "grasp #" << i << ", score: " << scores[i] << "\n";
//
//    if (scores[i] >= min_score_diff_)
//    {
//      std::cout << " grasp #" << i << ", score: " << scores[i] << "\n";
//      valid_grasps.push_back(grasp_list[i]);
//      valid_grasps[valid_grasps.size() - 1].setScore(scores[i]);
//      valid_grasps[valid_grasps.size() - 1].setFullAntipodal(true);
//    }
//  }
//  std::cout << "Found " << valid_grasps.size() << " grasps with a score >= " << min_score_diff_ << "\n";

  std::cout << "Total classification time: " << omp_get_wtime() - t0 << std::endl;

  if (plot_valid_grasps_)
  {
    Plot plotter;
    const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
    plotter.plotFingers3D(valid_grasps, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
      params.finger_width_, params.hand_depth_, params.hand_height_);
  }

  return selected_grasps;
}


std::vector<GraspSet> GraspDetector::filterGraspsWorkspace(const std::vector<GraspSet>& hand_set_list,
  const std::vector<double>& workspace)
{
  int remaining = 0;
  std::vector<GraspSet> hand_set_list_out;

  for (int i = 0; i < hand_set_list.size(); i++)
  {
    const std::vector<Grasp>& hands = hand_set_list[i].getHypotheses();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid = hand_set_list[i].getIsValid();

    for (int j = 0; j < hands.size(); j++)
    {
      if (is_valid(j))
      {
        double half_width = 0.5 * outer_diameter_;
        Eigen::Vector3d left_bottom = hands[j].getGraspBottom() + half_width * hands[j].getBinormal();
        Eigen::Vector3d right_bottom = hands[j].getGraspBottom() - half_width * hands[j].getBinormal();
        Eigen::Vector3d left_top = hands[j].getGraspTop() + half_width * hands[j].getBinormal();
        Eigen::Vector3d right_top = hands[j].getGraspTop() - half_width * hands[j].getBinormal();
        Eigen::Vector3d approach = hands[j].getGraspBottom() - 0.05 * hands[j].getApproach();
        Eigen::VectorXd x(5), y(5), z(5);
        x << left_bottom(0), right_bottom(0), left_top(0), right_top(0), approach(0);
        y << left_bottom(1), right_bottom(1), left_top(1), right_top(1), approach(1);
        z << left_bottom(2), right_bottom(2), left_top(2), right_top(2), approach(2);
        double aperture = hands[j].getGraspWidth();

        if (aperture >= min_aperture_ && aperture <= max_aperture_ // make sure the object fits into the hand
          && x.minCoeff() >= workspace[0] && x.maxCoeff() <= workspace[1] // avoid grasping outside the x-workspace
          && y.minCoeff() >= workspace[2] && y.maxCoeff() <= workspace[3] // avoid grasping outside the y-workspace
          && z.minCoeff() >= workspace[4] && z.maxCoeff() <= workspace[5]) // avoid grasping outside the z-workspace
        {
          is_valid(j) = true;
          remaining++;
        }
        else
        {
          is_valid(j) = false;
        }
      }
    }

    if (is_valid.any())
    {
      hand_set_list_out.push_back(hand_set_list[i]);
      hand_set_list_out[hand_set_list_out.size() - 1].setIsValid(is_valid);
    }
  }

  ROS_INFO_STREAM("# grasps within workspace and gripper width: " << remaining);

  return hand_set_list_out;
}


std::vector<GraspSet> GraspDetector::filterSideGraspsCloseToTable(const std::vector<GraspSet>& hand_set_list)
{
  const double APPROACH_LENGTH = 0.05;

  int remaining = 0;
  std::vector<GraspSet> hand_set_list_out;
  Eigen::Vector3d vert_axis_vec;
  vert_axis_vec << vert_axis_[0], vert_axis_[1], vert_axis_[2];

  for (int i = 0; i < hand_set_list.size(); i++)
  {
    const std::vector<Grasp>& hands = hand_set_list[i].getHypotheses();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid = hand_set_list[i].getIsValid();

    for (int j = 0; j < hands.size(); j++)
    {
      if (is_valid(j))
      {
        //double angle = fabs(vert_axis_vec.transpose() * hands[j].getApproach());
        //double dist = fabs((hands[j].getGraspBottom() - APPROACH_LENGTH*hands[j].getApproach())(2)) - table_height_;

        double half_width = 0.5 * outer_diameter_;
        Eigen::Vector3d left_top = hands[j].getGraspTop() + half_width * hands[j].getBinormal();
        Eigen::Vector3d right_top = hands[j].getGraspTop() - half_width * hands[j].getBinormal();

        // This is a side grasps that is too close to the table.
        //if (angle > angle_thresh_ && dist < table_thresh_)
        if (left_top.z() < table_height_ || right_top.z() < table_height_)
        {
          is_valid(j) = false;
        }
        else
        {
          is_valid(j) = true;
          remaining++;
        }
      }
    }

    if (is_valid.any())
    {
      hand_set_list_out.push_back(hand_set_list[i]);
      hand_set_list_out[hand_set_list_out.size() - 1].setIsValid(is_valid);
    }
  }

  ROS_INFO_STREAM("# grasps that are not too close to the table: " << remaining);

  return hand_set_list_out;
}


std::vector<GraspSet> GraspDetector::filterHalfAntipodal(const std::vector<GraspSet>& hand_set_list)
{
  int remaining = 0;
  std::vector<GraspSet> hand_set_list_out;

  for (int i = 0; i < hand_set_list.size(); i++)
  {
    const std::vector<Grasp>& hands = hand_set_list[i].getHypotheses();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid = hand_set_list[i].getIsValid();

    for (int j = 0; j < hands.size(); j++)
    {
      if (is_valid(j))
      {
        if (!hands[j].isHalfAntipodal() || hands[j].isFullAntipodal())
        {
          is_valid(j) = true;
          remaining++;
        }
        else
        {
          is_valid(j) = false;
        }
      }
    }

    if (is_valid.any())
    {
      hand_set_list_out.push_back(hand_set_list[i]);
      hand_set_list_out[hand_set_list_out.size() - 1].setIsValid(is_valid);
    }
  }

  ROS_INFO_STREAM("# grasps that are not half-antipodal: " << remaining);

  return hand_set_list_out;
}


std::vector<Grasp> GraspDetector::extractHypotheses(const std::vector<GraspSet>& hand_set_list)
{
  std::vector<Grasp> hands_out;
  hands_out.resize(0);

  for (int i = 0; i < hand_set_list.size(); i++)
  {
    const std::vector<Grasp>& hands = hand_set_list[i].getHypotheses();

    for (int j = 0; j < hands.size(); j++)
    {
      if (hand_set_list[i].getIsValid()(j))
      {
        hands_out.push_back(hands[j]);
      }
    }
  }

  return hands_out;
}


void GraspDetector::extractGraspsAndImages(const std::vector<GraspSet>& hand_set_list,
  const std::vector<cv::Mat>& images, std::vector<Grasp>& grasps_out, std::vector<cv::Mat>& images_out)
{
  grasps_out.resize(0);
  images_out.resize(0);
  int num_orientations = hand_set_list[0].getHypotheses().size();

  for (int i = 0; i < hand_set_list.size(); i++)
  {
    const std::vector<Grasp>& hands = hand_set_list[i].getHypotheses();

    for (int j = 0; j < hands.size(); j++)
    {
      if (hand_set_list[i].getIsValid()(j))
      {
        grasps_out.push_back(hands[j]);
        images_out.push_back(images[i * num_orientations + j]);
      }
    }
  }
}


std::vector<Grasp> GraspDetector::findClusters(const std::vector<Grasp>& grasps)
{
  return clustering_->findClusters(grasps);
}
