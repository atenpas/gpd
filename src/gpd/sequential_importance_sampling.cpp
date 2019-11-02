#include <gpd/sequential_importance_sampling.h>

#include <random>

namespace gpd {

// methods for sampling from a set of Gaussians
const int SUM_OF_GAUSSIANS = 0;
const int MAX_OF_GAUSSIANS = 1;

SequentialImportanceSampling::SequentialImportanceSampling(
    const std::string &config_filename) {
  // Read parameters from configuration file.
  util::ConfigFile config_file(config_filename);
  config_file.ExtractKeys();

  num_init_samples_ = config_file.getValueOfKey<int>("num_init_samples", 50);
  num_iterations_ = config_file.getValueOfKey<int>("num_iterations", 5);
  num_samples_ =
      config_file.getValueOfKey<int>("num_samples_per_iteration", 50);
  prob_rand_samples_ =
      config_file.getValueOfKey<double>("prob_rand_samples", 0.3);
  radius_ = config_file.getValueOfKey<double>("standard_deviation", 0.02);
  sampling_method_ =
      config_file.getValueOfKey<int>("sampling_method", SUM_OF_GAUSSIANS);
  min_score_ = config_file.getValueOfKey<double>("min_score", 0);

  num_threads_ = config_file.getValueOfKey<int>("num_threads", 1);

  workspace_ =
      config_file.getValueOfKeyAsStdVectorDouble("workspace", "-1 1 -1 1 -1 1");
  workspace_grasps_ = config_file.getValueOfKeyAsStdVectorDouble(
      "workspace_grasps", "-1 1 -1 1 -1 1");

  filter_approach_direction_ =
      config_file.getValueOfKey<bool>("filter_approach_direction", false);
  std::vector<double> approach =
      config_file.getValueOfKeyAsStdVectorDouble("direction", "1 0 0");
  direction_ << approach[0], approach[1], approach[2];
  thresh_rad_ = config_file.getValueOfKey<double>("thresh_rad", 2.3);

  visualize_rounds_ =
      config_file.getValueOfKey<bool>("visualize_rounds", false);
  visualize_steps_ = config_file.getValueOfKey<bool>("visualize_steps", false);
  visualize_results_ =
      config_file.getValueOfKey<bool>("visualize_results", false);

  grasp_detector_ = std::make_unique<GraspDetector>(config_filename);

  int min_inliers = config_file.getValueOfKey<int>("min_inliers", 1);
  clustering_ = std::make_unique<Clustering>(min_inliers);
}

std::vector<std::unique_ptr<candidate::Hand>>
SequentialImportanceSampling::detectGrasps(util::Cloud &cloud) {
  if (cloud.getCloudOriginal()->size() == 0) {
    printf("Error: Point cloud is empty!");
    std::vector<std::unique_ptr<candidate::Hand>> grasps(0);
    return grasps;
  }

  double t0 = omp_get_wtime();

  const candidate::HandGeometry &hand_geom =
      grasp_detector_->getHandSearchParameters().hand_geometry_;

  util::Plot plotter(
      grasp_detector_->getHandSearchParameters().hand_axes_.size(),
      grasp_detector_->getHandSearchParameters().num_orientations_);

  // 1. Find initial grasp hypotheses.
  cloud.subsample(num_init_samples_);
  if (visualize_steps_) {
    plotter.plotSamples(cloud.getSampleIndices(), cloud.getCloudProcessed());
  }
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list =
      grasp_detector_->generateGraspCandidates(cloud);
  printf("Initially detected grasp candidates: %zu\n", hand_set_list.size());
  if (hand_set_list.size() == 0) {
    std::vector<std::unique_ptr<candidate::Hand>> grasps(0);
    return grasps;
  }

  hand_set_list =
      grasp_detector_->filterGraspsWorkspace(hand_set_list, workspace_grasps_);
  printf("Grasps within workspace: %zu", hand_set_list.size());

  if (filter_approach_direction_) {
    hand_set_list = grasp_detector_->filterGraspsDirection(
        hand_set_list, direction_, thresh_rad_);
    if (visualize_rounds_) {
      plotter.plotFingers3D(hand_set_list, cloud.getCloudOriginal(),
                            "Filtered Grasps (Approach)", hand_geom);
    }
  }

  if (visualize_rounds_) {
    plotter.plotFingers3D(hand_set_list, cloud.getCloudOriginal(),
                          "Initial Grasps", hand_geom);
  }

  int num_rand_samples = prob_rand_samples_ * num_samples_;
  int num_gauss_samples = num_samples_ - num_rand_samples;
  double sigma = radius_;
  Eigen::Matrix3d diag_sigma = Eigen::Matrix3d::Zero();
  diag_sigma.diagonal() << sigma, sigma, sigma;
  Eigen::Matrix3d inv_sigma = diag_sigma.inverse();
  double term = 1.0 / sqrt(pow(2.0 * M_PI, 3.0) * pow(sigma, 3.0));
  Eigen::Matrix3Xd samples(3, num_samples_);

  // 2. Find grasp hypotheses using importance sampling.
  for (int i = 0; i < num_iterations_; i++) {
    std::cout << i << " " << num_gauss_samples << std::endl;

    // 2.1 Draw samples close to existing affordances.
    if (this->sampling_method_ == SUM_OF_GAUSSIANS) {
      drawSamplesFromSumOfGaussians(hand_set_list, sigma, num_gauss_samples,
                                    samples);
    } else if (this->sampling_method_ == MAX_OF_GAUSSIANS) {
      drawSamplesFromMaxOfGaussians(hand_set_list, sigma, num_gauss_samples,
                                    samples, term);
    }

    // 2.2 Draw random samples.
    drawUniformSamples(cloud, num_rand_samples, num_samples_ - num_rand_samples,
                       samples);

    // 2.3 Evaluate grasp hypotheses at <samples>.
    cloud.setSamples(samples);
    std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_new =
        grasp_detector_->generateGraspCandidates(cloud);

    hand_set_list_new = grasp_detector_->filterGraspsWorkspace(
        hand_set_list_new, workspace_grasps_);

    if (filter_approach_direction_) {
      hand_set_list_new = grasp_detector_->filterGraspsDirection(
          hand_set_list_new, direction_, thresh_rad_);
      if (visualize_steps_) {
        plotter.plotFingers3D(hand_set_list_new, cloud.getCloudOriginal(),
                              "Filtered Grasps (Approach)", hand_geom);
      }
    }

    if (visualize_rounds_) {
      plotter.plotSamples(samples, cloud.getCloudProcessed());
      plotter.plotFingers3D(hand_set_list_new, cloud.getCloudOriginal(),
                            "New Grasps", hand_geom);
    }

    hand_set_list.insert(hand_set_list.end(),
                         std::make_move_iterator(hand_set_list_new.begin()),
                         std::make_move_iterator(hand_set_list_new.end()));

    printf("Added %zu grasp candidates in round %d. Total: %zu.\n",
           hand_set_list_new.size(), i, hand_set_list.size());
  }

  if (visualize_steps_) {
    plotter.plotFingers3D(hand_set_list, cloud.getCloudOriginal(),
                          "Grasp Candidates", hand_geom);
  }

  // 3. Classify the grasps.
  std::vector<std::unique_ptr<candidate::Hand>> valid_grasps;
  valid_grasps =
      grasp_detector_->pruneGraspCandidates(cloud, hand_set_list, min_score_);
  printf("Valid grasps: %zu\n", valid_grasps.size());
  if (visualize_steps_) {
    plotter.plotFingers3D(valid_grasps, cloud.getCloudOriginal(),
                          "Valid Grasps", hand_geom);
  }

  // 4. Cluster the grasps.
  if (clustering_->getMinInliers() > 0) {
    valid_grasps = clustering_->findClusters(valid_grasps);
  }
  printf("Final result: found %zu grasps.\n", valid_grasps.size());
  printf("Total runtime: %3.4fs\n.\n", omp_get_wtime() - t0);

  if (visualize_results_ || visualize_steps_) {
    plotter.plotFingers3D(valid_grasps, cloud.getCloudOriginal(), "Clusters",
                          hand_geom);
  }

  return valid_grasps;
}

void SequentialImportanceSampling::drawSamplesFromSumOfGaussians(
    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_sets,
    double sigma, int num_gauss_samples, Eigen::Matrix3Xd &samples_out) {
  static std::random_device rd{};
  static std::mt19937 gen{rd()};
  static std::normal_distribution<double> distr{0.0, sigma};
  for (std::size_t j = 0; j < num_gauss_samples; j++) {
    int idx = rand() % hand_sets.size();
    Eigen::Vector3d rand_vec;
    rand_vec << distr(gen), distr(gen), distr(gen);
    samples_out.col(j) = hand_sets[idx]->getSample() + rand_vec;
  }
}

void SequentialImportanceSampling::drawSamplesFromMaxOfGaussians(
    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_sets,
    double sigma, int num_gauss_samples, Eigen::Matrix3Xd &samples_out,
    double term) {
  int j = 0;
  static std::random_device rd{};
  static std::mt19937 gen{rd()};
  static std::normal_distribution<double> distr{0.0, sigma};

  // Draw samples using rejection sampling.
  while (j < num_gauss_samples) {
    int idx = rand() % hand_sets.size();
    Eigen::Vector3d rand_vec;
    rand_vec << distr(gen), distr(gen), distr(gen);
    Eigen::Vector3d x = hand_sets[idx]->getSample() + rand_vec;

    double maxp = 0;
    for (std::size_t k = 0; k < hand_sets.size(); k++) {
      double p = (x - hand_sets[k]->getSample()).transpose() *
                 (x - hand_sets[k]->getSample());
      p = term * exp((-1.0 / (2.0 * sigma)) * p);
      if (p > maxp) {
        maxp = p;
      }
    }

    double p = (x - hand_sets[idx]->getSample()).transpose() *
               (x - hand_sets[idx]->getSample());
    p = term * exp((-1.0 / (2.0 * sigma)) * p);
    if (p >= maxp) {
      samples_out.col(j) = x;
      j++;
    }
  }
}

void SequentialImportanceSampling::drawUniformSamples(
    const util::Cloud &cloud, int num_samples, int start_idx,
    Eigen::Matrix3Xd &samples) {
  int i = 0;
  while (i < num_samples) {
    int idx;
    Eigen::Vector3d sample;
    if (cloud.getSampleIndices().size() > 0) {
      int idx = cloud.getSampleIndices()[std::rand() %
                                         cloud.getSampleIndices().size()];
      sample = cloud.getCloudProcessed()
                   ->points[idx]
                   .getVector3fMap()
                   .cast<double>();
    } else if (cloud.getSamples().size() > 0) {
      int idx = std::rand() % cloud.getSamples().size();
      sample = cloud.getSamples().col(idx);
    } else {
      int idx = std::rand() % cloud.getCloudProcessed()->points.size();
      sample = cloud.getCloudProcessed()
                   ->points[idx]
                   .getVector3fMap()
                   .cast<double>();
    }
    if (sample(0) >= workspace_[0] && sample(0) <= workspace_[1] &&
        sample(1) >= workspace_[2] && sample(1) <= workspace_[3] &&
        sample(2) >= workspace_[4] && sample(2) <= workspace_[5]) {
      samples.col(start_idx + i) = sample;
      i++;
    }
  }
}

}  // namespace gpd
