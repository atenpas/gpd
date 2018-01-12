#include <gpd/sequential_importance_sampling.h>


// methods for sampling from a set of Gaussians
const int SUM_OF_GAUSSIANS = 0;
const int MAX_OF_GAUSSIANS = 1;

// standard parameters
const int SequentialImportanceSampling::NUM_ITERATIONS = 5;
const int SequentialImportanceSampling::NUM_SAMPLES = 50;
const int SequentialImportanceSampling::NUM_INIT_SAMPLES = 50;
const double SequentialImportanceSampling::PROB_RAND_SAMPLES = 0.3;
const double SequentialImportanceSampling::RADIUS = 0.02;
const bool SequentialImportanceSampling::VISUALIZE_STEPS = false;
const bool SequentialImportanceSampling::VISUALIZE_RESULTS = true;
const int SequentialImportanceSampling::SAMPLING_METHOD = SUM_OF_GAUSSIANS;


SequentialImportanceSampling::SequentialImportanceSampling(ros::NodeHandle& node)
{
  node.param("num_init_samples", num_init_samples_, NUM_INIT_SAMPLES);
  node.param("num_iterations", num_iterations_, NUM_ITERATIONS);
  node.param("num_samples_per_iteration", num_samples_, NUM_SAMPLES);
  node.param("prob_rand_samples", prob_rand_samples_, PROB_RAND_SAMPLES);
  node.param("std", radius_, RADIUS);
  node.param("sampling_method", sampling_method_, SAMPLING_METHOD);
  node.param("visualize_rounds", visualize_rounds_, false);
  node.param("visualize_steps", visualize_steps_, VISUALIZE_STEPS);
  node.param("visualize_results", visualize_results_, VISUALIZE_RESULTS);
  node.param("filter_grasps", filter_grasps_, false);
  node.param("num_threads", num_threads_, 1);

  node.getParam("workspace", workspace_);
  node.getParam("workspace_grasps", workspace_grasps_);

  grasp_detector_ = new GraspDetector(node);
}


std::vector<Grasp> SequentialImportanceSampling::detectGrasps(const CloudCamera& cloud_cam_in)
{
  // Check if the point cloud is empty.
  if (cloud_cam_in.getCloudOriginal()->size() == 0)
  {
    ROS_INFO("Point cloud is empty!");
    std::vector<Grasp> grasps(0);
    return grasps;
  }

  double t0 = omp_get_wtime();
  CloudCamera cloud_cam = cloud_cam_in;
  Plot plotter;
  const PointCloudRGB::Ptr& cloud = cloud_cam.getCloudProcessed();

  // 1. Find initial grasp hypotheses.
  cloud_cam.subsampleUniformly(num_init_samples_);
  std::vector<GraspSet> hand_set_list = grasp_detector_->generateGraspCandidates(cloud_cam);
  std::cout << "Initially detected " << hand_set_list.size() << " grasp hypotheses" << std::endl;
  if (hand_set_list.size() == 0)
  {
    std::vector<Grasp> empty_grasps(0);
    return empty_grasps;
  }

  // Filter grasps outside of workspace and robot hand dimensions.
  std::vector<GraspSet> filtered_candidates;
  if (filter_grasps_)
  {
    grasp_detector_->filterGraspsWorkspace(hand_set_list, workspace_grasps_);
    ROS_INFO_STREAM("Number of grasps within workspace: " << filtered_candidates.size());
  }

  if (visualize_rounds_)
  {
    plotter.plotFingers(hand_set_list, cloud_cam.getCloudOriginal(), "Initial Grasps");
  }

  // Classify the grasps.
//  classified_grasps = grasp_detector_->classifyGraspCandidates(cloud_cam, hands);
//  std::cout << "Predicted " << filtered_candidates.size() << " valid grasps.\n";
//
//  if (visualize_steps_)
//  {
//    plotter.plotFingers(classified_grasps, cloud_cam.getCloudOriginal(), "Initial Grasps");
//  }
//
//  if (classified_grasps.size() == 0)
//  {
//    return classified_grasps;
//  }

//  if (classified_grasps.size() < 5)
//  {
//    classified_grasps = hands;
//  }

  // 2. Create random generator for normal distribution.
  int num_rand_samples = prob_rand_samples_ * num_samples_;
  int num_gauss_samples = num_samples_ - num_rand_samples;
  double sigma = radius_;
  Eigen::Matrix3d diag_sigma = Eigen::Matrix3d::Zero();
  diag_sigma.diagonal() << sigma, sigma, sigma;
  Eigen::Matrix3d inv_sigma = diag_sigma.inverse();
  double term = 1.0 / sqrt(pow(2.0 * M_PI, 3.0) * pow(sigma, 3.0));
  boost::mt19937 *rng = new boost::mt19937();
  rng->seed(time(NULL));
  boost::normal_distribution<> distribution(0.0, 1.0);
  boost::variate_generator<boost::mt19937, boost::normal_distribution<> > generator(*rng, distribution);
  Eigen::Matrix3Xd samples(3, num_samples_);
//  std::vector<GraspHypothesis> hands_new = hands;

  // 3. Find grasp hypotheses using importance sampling.
  for (int i = 0; i < num_iterations_; i++)
  {
    std::cout << i << " " << num_gauss_samples << std::endl;

    // 3.1 Draw samples close to affordances (importance sampling).
    if (this->sampling_method_ == SUM_OF_GAUSSIANS)
    {
      drawSamplesFromSumOfGaussians(hand_set_list, generator, sigma, num_gauss_samples, samples);
    }
    else if (this->sampling_method_ == MAX_OF_GAUSSIANS) // max of Gaussians
    {
      drawSamplesFromMaxOfGaussians(hand_set_list, generator, sigma, num_gauss_samples, samples, term);
    }
    else
    {
//      drawWeightedSamples(hand_set_list, generator, sigma, num_gauss_samples, samples);
    }

    // 3.2 Draw random samples.
    for (int j = num_samples_ - num_rand_samples; j < num_samples_; j++)
    {
      int r = std::rand() % cloud->points.size();
//      while (!pcl::isFinite((*cloud)[r])
//          || !this->affordances.isPointInWorkspace(cloud->points[r].x, cloud->points[r].y, cloud->points[r].z))
//        r = std::rand() % cloud->points.size();
      samples.col(j) = cloud->points[r].getVector3fMap().cast<double>();
    }

    // 3.3 Evaluate grasp hypotheses at <samples>.
    cloud_cam.setSamples(samples);
    std::vector<GraspSet> hand_set_list_new = grasp_detector_->generateGraspCandidates(cloud_cam);

    if (filter_grasps_)
    {
      grasp_detector_->filterGraspsWorkspace(hand_set_list_new, workspace_grasps_);
      ROS_INFO_STREAM("Number of grasps within gripper width range and workspace: " << hand_set_list_new.size());
    }

    hand_set_list.insert(hand_set_list.end(), hand_set_list_new.begin(), hand_set_list_new.end());

    if (visualize_rounds_)
    {
      plotter.plotSamples(samples, cloud);
      plotter.plotFingers(hand_set_list_new, cloud_cam.getCloudProcessed(), "New Grasps");
    }

    std::cout << "Added: " << hand_set_list_new.size() << ", total: " << hand_set_list.size()
      << " grasp candidate sets in round " << i << std::endl;
  }
  if (visualize_steps_)
  {
    plotter.plotFingers(hand_set_list, cloud_cam.getCloudOriginal(), "Grasp Candidates");
  }

  // Classify the grasps.
  std::vector<Grasp> valid_grasps;
  valid_grasps = grasp_detector_->classifyGraspCandidates(cloud_cam, hand_set_list);
  std::cout << "Predicted " << valid_grasps.size() << " valid grasps.\n";
  if (visualize_steps_)
  {
    plotter.plotFingers(valid_grasps, cloud_cam.getCloudOriginal(), "Valid Grasps");
  }

  // 4. Cluster the grasps.
  valid_grasps = grasp_detector_->findClusters(valid_grasps);
  std::cout << "Final result: found " << valid_grasps.size() << " clusters.\n";
  std::cout << "Total runtime: " << omp_get_wtime() - t0 << " sec.\n";
  
  if (visualize_results_ || visualize_steps_)
  {
    plotter.plotFingers(valid_grasps, cloud_cam.getCloudOriginal(), "Clusters");
  }

  // Remove grasps that are very close in position and approach direction.
//  std::set<Grasp, compareGraspPositions> unique_grasps;
//  for (int i = 0; i < classified_grasps.size(); ++i)
//  {
//    unique_grasps.insert(classified_grasps[i]);
//  }
//  std::cout << "Found " << unique_grasps.size() << " unique clusters.\n";
//
//  classified_grasps.resize(unique_grasps.size());
//  std::copy(unique_grasps.begin(), unique_grasps.end(), classified_grasps.begin());

  return valid_grasps;
}


void SequentialImportanceSampling::preprocessPointCloud(CloudCamera& cloud_cam)
{
  std::cout << "Processing cloud with: " << cloud_cam.getCloudOriginal()->size() << " points.\n";
  cloud_cam.filterWorkspace(workspace_);
  std::cout << "After workspace filtering: " << cloud_cam.getCloudProcessed()->size() << " points left.\n";
  cloud_cam.voxelizeCloud(0.003);
  std::cout << "After voxelizing: " << cloud_cam.getCloudProcessed()->size() << " points left.\n";
  cloud_cam.calculateNormals(num_threads_);
}


void SequentialImportanceSampling::drawSamplesFromSumOfGaussians(const std::vector<GraspSet>& hands,
  Gaussian& generator, double sigma, int num_gauss_samples, Eigen::Matrix3Xd& samples_out)
{
  for (std::size_t j = 0; j < num_gauss_samples; j++)
  {
    int idx = rand() % hands.size();
    samples_out(0, j) = hands[idx].getSample()(0) + generator() * sigma;
    samples_out(1, j) = hands[idx].getSample()(1) + generator() * sigma;
    samples_out(2, j) = hands[idx].getSample()(2) + generator() * sigma;
  }
}


void SequentialImportanceSampling::drawSamplesFromMaxOfGaussians(const std::vector<GraspSet>& hands,
  Gaussian& generator, double sigma, int num_gauss_samples, Eigen::Matrix3Xd& samples_out, double term)
{
  int j = 0;
  while (j < num_gauss_samples) // draw samples using rejection sampling
  {
    int idx = rand() % hands.size();
    Eigen::Vector3d x;
    x(0) = hands[idx].getSample()(0) + generator() * sigma;
    x(1) = hands[idx].getSample()(1) + generator() * sigma;
    x(2) = hands[idx].getSample()(2) + generator() * sigma;

    double maxp = 0;
    for (std::size_t k = 0; k < hands.size(); k++)
    {
      double p = (x - hands[k].getSample()).transpose() * (x - hands[k].getSample());
      p = term * exp((-1.0 / (2.0 * sigma)) * p);
      if (p > maxp)
        maxp = p;
    }

    double p = (x - hands[idx].getSample()).transpose() * (x - hands[idx].getSample());
    p = term * exp((-1.0 / (2.0 * sigma)) * p);
    if (p >= maxp)
    {
      samples_out.col(j) = x;
      j++;
    }
  }
}


void SequentialImportanceSampling::drawWeightedSamples(const std::vector<Grasp>& hands, Gaussian& generator,
  double sigma, int num_gauss_samples, Eigen::Matrix3Xd& samples_out)
{
  Eigen::VectorXd scores(hands.size());
  double sum = 0.0;
  for (int i = 0; i < hands.size(); i++)
  {
    scores(i) = hands[i].getScore();
    sum += scores(i);
  }

  for (int i = 0; i < hands.size(); i++)
  {
    scores(i) /= sum;
  }

  boost::mt19937 *rng = new boost::mt19937();
  rng->seed(time(NULL));
  boost::uniform_real<> distribution(0.0, 1.0);
  boost::variate_generator<boost::mt19937, boost::uniform_real<> > uniform_generator(*rng, distribution);
//  std::cout << "scores\n" << scores << std::endl;

  for (int i = 0; i < num_gauss_samples; i++)
  {
    double r = uniform_generator();
    double x = 0.0;
    int idx = -1;

    for (int j = 0; j < scores.size(); j++)
    {
      x += scores(j);
      if (r < x)
      {
        idx = j;
        break;
      }
    }

    if (idx > -1)
    {
//      std::cout << "r: " << r << ", idx: " << idx << std::endl;
      samples_out(0,i) = hands[idx].getGraspSurface()(0) + generator() * sigma;
      samples_out(1,i) = hands[idx].getGraspSurface()(1) + generator() * sigma;
      samples_out(2,i) = hands[idx].getGraspSurface()(2) + generator() * sigma;
    }
    else
      std::cout << "Error: idx is " << idx << std::endl;
  }
}
