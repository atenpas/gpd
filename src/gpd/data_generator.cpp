#include "gpd/data_generator.h"


DataGenerator::DataGenerator(DataGenerationParameters& param)
{
  param_ = param;
  // Create object to generate grasp candidates.
  candidates_generator_ = new CandidatesGenerator(param_.generator_params, param_.hand_search_params);
  // Create object to create grasp images from grasp candidates (used for classification).
  learning_ = new Learning(param_.image_params, param_.num_threads, param_.num_orientations, false, param_.remove_plane);
}


DataGenerator::~DataGenerator()
{
  delete candidates_generator_;
  delete learning_;
}

CloudCamera DataGenerator::loadCloudCameraFromFile()
{
  return CloudCamera(param_.filename, param_.view_points);
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


void DataGenerator::createTrainingData()
{
  int store_step = 1;
  int max_grasps_per_view = 1000;
  Eigen::VectorXi test_views(5);
  test_views << 2, 5, 8, 13, 16;

  std::vector<std::string> objects = loadObjectNames(param_.objects_file_location);
  std::vector<int> positives_list, negatives_list;
  std::vector<Instance> train_data, test_data;
  train_data.reserve(store_step * param_.num_views * 100);
  test_data.reserve(store_step * param_.num_views * 100);

  for (int i = 0; i < objects.size(); i++)
  {
    printf("===> (%d) Generating images for object: %s\n", i, objects[i].c_str());

    // Load mesh for ground truth.
    std::string prefix = param_.data_root + objects[i];
    CloudCamera mesh_cloud_cam = loadMesh(prefix + "_gt.pcd", prefix + "_gt_normals.csv");

    for (int j = 0; j < param_.num_views; j++)
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

      if (param_.plot_grasps)
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
  storeLMDB(train_data, param_.output_root + "train_lmdb");
  storeLMDB(test_data, param_.output_root + "test_lmdb");
  std::cout << "Wrote data to training and test LMDBs\n";
}
