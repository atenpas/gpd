#include "gpd/grasp_detector.h"


GraspDetector::GraspDetector(GraspDetectionParameters& param)
{
  Eigen::initParallel();

  param_ = param;

  outer_diameter_ = param_.hand_search_params.hand_outer_diameter_;

  candidates_generator_ = new CandidatesGenerator(param_.generator_params, param_.hand_search_params);

  classifier_ = Classifier::create(param_.model_file_, param_.weights_file_, static_cast<Classifier::Device>(param_.device_));

  // Create object to create grasp images from grasp candidates (used for classification)
  learning_ = new Learning(param_.image_params, param_.hand_search_params.num_threads_,
    param_.hand_search_params.num_orientations_, false, param_.remove_plane_);

  min_aperture_ = param_.gripper_width_range_[0];
  max_aperture_ = param_.gripper_width_range_[1];

  clustering_ = new Clustering(param_.min_inliers_);
  cluster_grasps_ = param_.min_inliers_ > 0 ? true : false;

}


std::vector<Grasp> GraspDetector::detectGrasps(const CloudCamera& cloud_cam)
{
  std::vector<Grasp> selected_grasps(0);

  // Check if the point cloud is empty.
  if (cloud_cam.getCloudOriginal()->size() == 0)
  {
    std::cout << "Point cloud is empty!\n";
    return selected_grasps;
  }

  Plot plotter;

  // Plot samples/indices.
  if (param_.plot_samples_)
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

  if (param_.plot_normals_)
  {
    std::cout << "Plotting normals for different camera sources\n";
    plotter.plotNormals(cloud_cam);
  }

  // 1. Generate grasp candidates.
  std::vector<GraspSet> candidates = generateGraspCandidates(cloud_cam);
  std::cout << "Generated " << candidates.size() << " grasp candidate sets.\n";
  if (candidates.size() == 0)
  {
    return selected_grasps;
  }

  // 2.1 Prune grasp candidates based on min. and max. robot hand aperture and fingers below table surface.
  if (param_.filter_grasps_)
  {
    candidates = filterGraspsWorkspace(candidates, param_.workspace_);

    if (param_.plot_filtered_grasps_)
    {
      const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
      plotter.plotFingers3D(candidates, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
        params.finger_width_, params.hand_depth_, params.hand_height_);
    }
  }

  // 2.2 Filter half grasps.
  if (param_.filter_half_antipodal_)
  {
    candidates = filterHalfAntipodal(candidates);

    if (param_.plot_filtered_grasps_)
    {
      const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
      plotter.plotFingers3D(candidates, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
        params.finger_width_, params.hand_depth_, params.hand_height_);
    }
  }

  // 3. Classify each grasp candidate. (Note: switch from a list of hypothesis sets to a list of grasp hypotheses)
  std::vector<Grasp> valid_grasps = classifyGraspCandidates(cloud_cam, candidates);
  std::cout << "Predicted " << valid_grasps.size() << " valid grasps.\n";

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
    std::cout << "Found " << clustered_grasps.size() << " clusters.\n";
    if (clustered_grasps.size() <= 3)
    {
      std::cout << "Not enough clusters found! Using all grasps from previous step.\n";
      clustered_grasps = valid_grasps;
    }

    if (param_.plot_clusters_)
    {
      const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
      plotter.plotFingers3D(clustered_grasps, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
        params.finger_width_, params.hand_depth_, params.hand_height_);
    }
  }
  else
  {
    clustered_grasps = valid_grasps;
  }

  // 5. Select highest-scoring grasps.
  if (clustered_grasps.size() > param_.num_selected_)
  {
    std::cout << "Partial Sorting the grasps based on their score ... \n";
    std::partial_sort(clustered_grasps.begin(), clustered_grasps.begin() + param_.num_selected_, clustered_grasps.end(),
      isScoreGreater);
    selected_grasps.assign(clustered_grasps.begin(), clustered_grasps.begin() + param_.num_selected_);
  }
  else
  {
    std::cout << "Sorting the grasps based on their score ... \n";
    std::sort(clustered_grasps.begin(), clustered_grasps.end(), isScoreGreater);
    selected_grasps = clustered_grasps;
  }

  for (int i = 0; i < selected_grasps.size(); i++)
  {
    std::cout << "Grasp " << i << ": " << selected_grasps[i].getScore() << "\n";
  }

  std::cout << "Selected the " << selected_grasps.size() << " highest scoring grasps.\n";

  if (param_.plot_selected_grasps_)
  {
    const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
    plotter.plotFingers3D(selected_grasps, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
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
  std::vector<Grasp> grasp_list;
  int num_orientations = candidates[0].getHypotheses().size();

  // Create images in batches if required (less memory usage).
  if (param_.create_image_batches_)
  {
    int batch_size = classifier_->getBatchSize();
    int num_iterations = (int) ceil(candidates.size() * num_orientations / (double) batch_size);
    int step_size = (int) floor(batch_size / (double) num_orientations);
    std::cout << " num_iterations: " << num_iterations << ", step_size: " << step_size << "\n";

    // Process the grasp candidates in batches.
    for (int i = 0; i < num_iterations; i++)
    {
      std::cout << i << "\n";
      std::vector<GraspSet>::iterator start = candidates.begin() + i * step_size;
      std::vector<GraspSet>::iterator stop;
      if (i < num_iterations - 1)
      {
        stop = candidates.begin() + i * step_size + step_size;
      }
      else
      {
        stop = candidates.end();
      }

      std::vector<GraspSet> hand_set_sublist(start, stop);
      std::vector<cv::Mat> image_list = learning_->createImages(cloud_cam, hand_set_sublist);

      std::vector<Grasp> valid_grasps;
      std::vector<cv::Mat> valid_images;
      extractGraspsAndImages(candidates, image_list, valid_grasps, valid_images);

      std::vector<float> scores_sublist = classifier_->classifyImages(valid_images);
      scores.insert(scores.end(), scores_sublist.begin(), scores_sublist.end());
      grasp_list.insert(grasp_list.end(), valid_grasps.begin(), valid_grasps.end());
    }
  }
  else
  {
    // Create the grasp images.
    std::vector<cv::Mat> image_list = learning_->createImages(cloud_cam, candidates);
    std::cout << " Image creation time: " << omp_get_wtime() - t0 << std::endl;

    std::vector<Grasp> valid_grasps;
    std::vector<cv::Mat> valid_images;
    extractGraspsAndImages(candidates, image_list, valid_grasps, valid_images);

    // Classify the grasp images.
    double t0_prediction = omp_get_wtime();
    scores = classifier_->classifyImages(valid_images);
    grasp_list.assign(valid_grasps.begin(), valid_grasps.end());
    std::cout << " Prediction time: " << omp_get_wtime() - t0 << std::endl;
  }

  // Select grasps with a score of at least <min_score_diff_>.
  std::vector<Grasp> valid_grasps;

  for (int i = 0; i < grasp_list.size(); i++)
  {
    if (scores[i] >= param_.min_score_diff_)
    {
      std::cout << "grasp #" << i << ", score: " << scores[i] << "\n";
      valid_grasps.push_back(grasp_list[i]);
      valid_grasps[valid_grasps.size() - 1].setScore(scores[i]);
      valid_grasps[valid_grasps.size() - 1].setFullAntipodal(true);
    }
  }

  std::cout << "Found " << valid_grasps.size() << " grasps with a score >= " << param_.min_score_diff_ << "\n";
  std::cout << "Total classification time: " << omp_get_wtime() - t0 << std::endl;

  if (param_.plot_valid_grasps_)
  {
    Plot plotter;
    const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
    plotter.plotFingers3D(valid_grasps, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
      params.finger_width_, params.hand_depth_, params.hand_height_);
  }

  return valid_grasps;
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

  std::cout << "# grasps within workspace and gripper width: " << remaining << "\n";

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

  std::cout << "# grasps that are not half-antipodal: " << remaining << "\n";

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
