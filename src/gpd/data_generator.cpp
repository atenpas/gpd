#include "../../include/gpd/data_generator.h"


DataGenerator::DataGenerator(ros::NodeHandle& node)
{
  candidates_generator_ = createCandidatesGenerator(node);
  learning_ = createLearning(node);
}


DataGenerator::~DataGenerator()
{
  delete candidates_generator_;
  delete learning_;
}


CandidatesGenerator* DataGenerator::createCandidatesGenerator(ros::NodeHandle& node)
{
  // Create objects to store parameters
  CandidatesGenerator::Parameters generator_params;
  HandSearch::Parameters hand_search_params;

  // Read hand geometry parameters
  node.param("finger_width", hand_search_params.finger_width_, 0.01);
  node.param("hand_outer_diameter", hand_search_params.hand_outer_diameter_, 0.09);
  node.param("hand_depth", hand_search_params.hand_depth_, 0.06);
  node.param("hand_height", hand_search_params.hand_height_, 0.02);
  node.param("init_bite", hand_search_params.init_bite_, 0.015);

  // Read local hand search parameters
  node.param("nn_radius", hand_search_params.nn_radius_frames_, 0.01);
  node.param("num_orientations", hand_search_params.num_orientations_, 8);
  node.param("num_samples", hand_search_params.num_samples_, 500);
  node.param("num_threads", hand_search_params.num_threads_, 1);
  node.param("rotation_axis", hand_search_params.rotation_axis_, 2);

  // Read general parameters
  generator_params.num_samples_ = hand_search_params.num_samples_;
  generator_params.num_threads_ = hand_search_params.num_threads_;
  node.param("plot_candidates", generator_params.plot_grasps_, false);

  // Read preprocessing parameters
  node.param("remove_outliers", generator_params.remove_statistical_outliers_, true);
  node.param("voxelize", generator_params.voxelize_, true);
  node.getParam("workspace", generator_params.workspace_);

  // Read plotting parameters.
  generator_params.plot_grasps_ = false;
  node.param("plot_normals", generator_params.plot_normals_, false);

  // Create object to generate grasp candidates.
  return new CandidatesGenerator(generator_params, hand_search_params);
}


Learning* DataGenerator::createLearning(ros::NodeHandle& node)
{
  // Read grasp image parameters.
  Learning::ImageParameters image_params;
  node.param("image_outer_diameter", image_params.outer_diameter_, 0.09);
  node.param("image_depth", image_params.depth_, 0.06);
  node.param("image_height", image_params.height_, 0.02);
  node.param("image_size", image_params.size_, 60);
  node.param("image_num_channels", image_params.num_channels_, 15);

  // Read learning parameters.
  bool remove_plane;
  int num_orientations, num_threads;
  node.param("remove_plane_before_image_calculation", remove_plane, false);
  node.param("num_orientations", num_orientations, 8);
  node.param("num_threads", num_threads, 1);

  // Create object to create grasp images from grasp candidates (used for classification).
  return new Learning(image_params, num_threads, num_orientations, false, remove_plane);
}


CloudCamera DataGenerator::loadCloudCameraFromFile(ros::NodeHandle& node)
{
  // Set the position from which the camera sees the point cloud.
  std::vector<double> camera_position;
  node.getParam("camera_position", camera_position);
  Eigen::Matrix3Xd view_points(3,1);
  view_points << camera_position[0], camera_position[1], camera_position[2];

  // Load the point cloud from the file.
  std::string filename;
  node.param("cloud_file_name", filename, std::string(""));

  return CloudCamera(filename, view_points);
}


CloudCamera DataGenerator::loadMesh(const std::string& mesh_file_path, const std::string& normals_file_path)
{
  // Load mesh for ground truth.
  std::cout << " mesh_file_path: " << mesh_file_path << '\n';
  std::cout << " normals_file_path: " << normals_file_path << '\n';

  // Set the position from which the camera sees the point cloud.
  Eigen::Matrix3Xd view_points(3,1);
  view_points << 0.0, 0.0, 0.0;

  CloudCamera mesh_cloud_cam(mesh_file_path, view_points);

  // Load surface normals for the mesh.
  mesh_cloud_cam.setNormalsFromFile(normals_file_path);
  std::cout << "Loaded mesh with " << mesh_cloud_cam.getCloudProcessed()->size() << " points.\n";

  return mesh_cloud_cam;
}


std::vector<boost::filesystem::path> DataGenerator::loadPointCloudFiles(const std::string& cloud_folder)
{
  boost::filesystem::path path(cloud_folder);
  boost::filesystem::directory_iterator it(path);
  std::vector<boost::filesystem::path> files;

  while (it != boost::filesystem::directory_iterator())
  {
    const std::string& filepath = (*it).path().string();

    if (filepath.find("mesh") == std::string::npos && filepath.find(".pcd") != std::string::npos)
    {
      files.push_back((*it).path());
    }

    it++;
  }

  std::sort(files.begin(), files.end());

  return files;
}


std::vector<std::string> DataGenerator::loadObjectNames(const std::string& objects_file_location)
{
  std::ifstream in;
  in.open(objects_file_location.c_str());
  std::string line;
  std::vector<std::string> objects;

  while(std::getline(in, line))
  {
    std::stringstream lineStream(line);
    std::string object;
    std::getline(lineStream, object, '\n');
    std::cout << object << "\n";
    objects.push_back(object);
  }

  return objects;
}


bool DataGenerator::createGraspImages(CloudCamera& cloud_cam, std::vector<Grasp>& grasps_out,
  std::vector<cv::Mat>& images_out)
{
  // Preprocess the point cloud.
  candidates_generator_->preprocessPointCloud(cloud_cam);

  // Reverse direction of surface normals.
  cloud_cam.setNormals(cloud_cam.getNormals().array() * (-1.0));

  // Generate grasp candidates.
  std::vector<GraspSet> candidates = candidates_generator_->generateGraspCandidateSets(cloud_cam);

  if (candidates.size() == 0)
  {
    ROS_ERROR("No grasp candidates found!");
    return false;
  }

  // Create the grasp images.
  std::vector<cv::Mat> image_list = learning_->createImages(cloud_cam, candidates);
  grasps_out.resize(0);
  images_out.resize(0);
  learning_->extractGraspsAndImages(candidates, image_list, grasps_out, images_out);
  return true;
}


void DataGenerator::balanceInstances(int max_grasps_per_view, const std::vector<int>& positives_in,
  const std::vector<int>& negatives_in, std::vector<int>& positives_out, std::vector<int>& negatives_out)
{
  int end = 0;
  positives_out.resize(0);
  negatives_out.resize(0);

  // Number of positives is less than or equal to number of negatives
  if (positives_in.size() > 0 && positives_in.size() <= negatives_in.size())
  {
    end = std::min((int) positives_in.size(), max_grasps_per_view);
    positives_out.insert(positives_out.end(), positives_in.begin(), positives_in.begin() + end);
    negatives_out.insert(negatives_out.end(), negatives_in.begin(), negatives_in.begin() + end);
  }
  else
  {
    end = std::min((int) negatives_in.size(), max_grasps_per_view);
    negatives_out.insert(negatives_out.end(), negatives_in.begin(), negatives_in.begin() + end);

    if (positives_in.size() > negatives_in.size())
    {
      positives_out.insert(positives_out.end(), positives_in.begin(), positives_in.begin() + end);
    }
  }
}


void DataGenerator::addInstances(const std::vector<Grasp>& grasps, const std::vector<cv::Mat>& images,
  const std::vector<int>& positives, const std::vector<int>& negatives, std::vector<Instance>& dataset)
{
  for (int k = 0; k < positives.size(); k++)
  {
    int idx = positives[k];
    dataset.push_back(Instance(images[idx], grasps[idx].isFullAntipodal()));
  }

  for (int k = 0; k < negatives.size(); k++)
  {
    int idx = negatives[k];
    dataset.push_back(Instance(images[idx], grasps[idx].isFullAntipodal()));
  }
}


void DataGenerator::storeLMDB(const std::vector<Instance>& dataset, const std::string& file_location)
{
  // Create new database
  boost::scoped_ptr<caffe::db::DB> db(caffe::db::GetDB("lmdb"));
  db->Open(file_location, caffe::db::NEW);
  boost::scoped_ptr<caffe::db::Transaction> txn(db->NewTransaction());

  // Storing to db
  caffe::Datum datum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int i = 0; i < dataset.size(); i++)
  {
    caffe::CVMatToDatum(dataset[i].image_, &datum);
    datum.set_label(dataset[i].label_);

    std::string key_str = caffe::format_int(i, 8);
    printf("%s, size: %d x %d x %d, label: %d, %d\n", key_str.c_str(), datum.channels(), datum.height(), datum.width(), datum.label(), dataset[i].label_);
//    std::cout << key_str << ", " << datum.channels() << "x " datum.label() << ", " << dataset[i].label_ << "\n";

    // Put in db
    std::string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (count % 1000 == 0)
    {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }

    count++;
  }

  // write the last batch
  if (count % 1000 != 0)
  {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
}


void DataGenerator::createTrainingData(ros::NodeHandle& node)
{
  int store_step = 1;
  int max_grasps_per_view = 1000;
  Eigen::VectorXi test_views(5);
  test_views << 2, 5, 8, 13, 16;

  std::string data_root, objects_file_location, output_root;
  node.param("data_root", data_root, std::string(""));
  node.param("objects_file", objects_file_location, std::string(""));
  node.param("output_root", output_root, std::string(""));
  bool plot_grasps;
  node.param("plot_grasps", plot_grasps, false);
  int num_views;
  node.param("num_views", num_views, 20);

  std::vector<std::string> objects = loadObjectNames(objects_file_location);
  std::vector<int> positives_list, negatives_list;
  std::vector<Instance> train_data, test_data;
  train_data.reserve(store_step * num_views * 100);
  test_data.reserve(store_step * num_views * 100);

  for (int i = 0; i < objects.size(); i++)
  {
    printf("===> (%d) Generating images for object: %s\n", i, objects[i].c_str());

    // Load mesh for ground truth.
    std::string prefix = data_root + objects[i];
    CloudCamera mesh_cloud_cam = loadMesh(prefix + "_gt.pcd", prefix + "_gt_normals.csv");

    for (int j = 0; j < num_views; j++)
    {
      printf("===> Processing view %d\n", j + 1);

      // 1. Load point cloud.
      Eigen::Matrix3Xd view_points(3,1);
      view_points << 0.0, 0.0, 0.0; // TODO: Load camera position.
      CloudCamera cloud_cam(prefix + "_" + boost::lexical_cast<std::string>(j + 1) + ".pcd", view_points);

      // 2. Find grasps in point cloud.
      std::vector<Grasp> grasps;
      std::vector<cv::Mat> images;
      bool has_grasps = createGraspImages(cloud_cam, grasps, images);

      if (plot_grasps)
      {
        Plot plotter;
        plotter.plotNormals(cloud_cam.getCloudOriginal(), cloud_cam.getNormals());
        plotter.plotFingers(grasps, cloud_cam.getCloudOriginal(), "Grasps on view");
      }

      // 3. Evaluate grasps against ground truth (mesh).
      std::vector<Grasp> labeled_grasps = candidates_generator_->reevaluateHypotheses(mesh_cloud_cam, grasps);

      // 4. Split grasps into positives and negatives.
      std::vector<int> positives;
      std::vector<int> negatives;
      for (int k = 0; k < labeled_grasps.size(); k++)
      {
        if (labeled_grasps[k].isFullAntipodal())
          positives.push_back(k);
        else
          negatives.push_back(k);
      }
      printf("#grasps: %d, #positives: %d, #negatives: %d\n", (int) labeled_grasps.size(), (int) positives.size(),
        (int) negatives.size());

      // 5. Balance the number of positives and negatives.
      balanceInstances(max_grasps_per_view, positives, negatives, positives_list, negatives_list);
      printf("#positives: %d, #negatives: %d\n", (int) positives_list.size(), (int) negatives_list.size());

      // 6. Assign instances to training or test data.
      bool is_test_view = false;

      for (int k = 0; k < test_views.rows(); k++)
      {
        if (j == test_views(k))
        {
          is_test_view = true;
          break;
        }
      }

      if (is_test_view)
      {
        addInstances(labeled_grasps, images, positives_list, negatives_list, test_data);
        std::cout << "test view, # test data: " << test_data.size() << "\n";
      }
      else
      {
        addInstances(labeled_grasps, images, positives_list, negatives_list, train_data);
        std::cout << "train view, # train data: " << train_data.size() << "\n";
      }
    }
  }

  // Shuffle the data.
  std::random_shuffle(train_data.begin(), train_data.end());
  std::random_shuffle(test_data.begin(), test_data.end());

  // Store the grasp images and their labels in LMDBs.
  storeLMDB(train_data, output_root + "train_lmdb");
  storeLMDB(test_data, output_root + "test_lmdb");
  std::cout << "Wrote data to training and test LMDBs\n";
}
