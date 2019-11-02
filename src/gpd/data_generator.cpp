#include <gpd/data_generator.h>

#include <Eigen/StdVector>

namespace gpd {

const std::string DataGenerator::IMAGES_DS_NAME = "images";
const std::string DataGenerator::LABELS_DS_NAME = "labels";

DataGenerator::DataGenerator(const std::string &config_filename) {
  detector_ = std::make_unique<GraspDetector>(config_filename);

  // Read parameters from configuration file.
  util::ConfigFile config_file(config_filename);
  config_file.ExtractKeys();
  data_root_ = config_file.getValueOfKeyAsString("data_root", "");
  objects_file_location_ =
      config_file.getValueOfKeyAsString("objects_file_location", "");
  output_root_ = config_file.getValueOfKeyAsString("output_root", "");
  chunk_size_ = config_file.getValueOfKey<int>("chunk_size", 1000);
  max_in_memory_ = config_file.getValueOfKey<int>("max_in_memory", 100000);
  num_views_per_object_ =
      config_file.getValueOfKey<int>("num_views_per_object", 1);
  min_grasps_per_view_ =
      config_file.getValueOfKey<int>("min_grasps_per_view", 100);
  max_grasps_per_view_ =
      config_file.getValueOfKey<int>("max_grasps_per_view", 500);
  test_views_ =
      config_file.getValueOfKeyAsStdVectorInt("test_views", "3 7 11 15 19");
  num_threads_ = config_file.getValueOfKey<int>("num_threads", 1);
  num_samples_ = config_file.getValueOfKey<int>("num_samples", 500);
  remove_nans_ = config_file.getValueOfKey<bool>("remove_nans", true);
  voxel_size_views_ =
      config_file.getValueOfKey<double>("voxel_size_views", 0.003);
  normals_radius_ = config_file.getValueOfKey<double>("normals_radius", 0.03);
  reverse_mesh_normals_ =
      config_file.getValueOfKey<bool>("reverse_mesh_normals", true);
  reverse_view_normals_ =
      config_file.getValueOfKey<bool>("reverse_view_normals", true);

  printf("============ DATA GENERATION =================\n");
  std::cout << "data_root: " << data_root_ << "\n";
  std::cout << "objects_file_location: " << objects_file_location_ << "\n";
  std::cout << "output_root: " << output_root_ << "\n";
  std::cout << "num_views_per_object: " << num_views_per_object_ << "\n";
  std::cout << "min_grasps_per_view: " << min_grasps_per_view_ << "\n";
  std::cout << "max_grasps_per_view: " << max_grasps_per_view_ << "\n";
  std::cout << "test_views: ";
  for (int i = 0; i < test_views_.size(); i++) {
    std::cout << test_views_[i] << " ";
  }
  std::cout << "\n";
  printf("==============================================\n");

  printf("============ CLOUD PREPROCESSING =============\n");
  printf("remove_nans: %d\n", remove_nans_);
  printf("voxel_size_views: %.3f\n", voxel_size_views_);
  printf("normals_radius_: %.3f\n", normals_radius_);
  printf("reverse_mesh_normals: %d\n", reverse_mesh_normals_);
  printf("reverse_view_normals: %d\n", reverse_view_normals_);
  printf("==============================================\n");

  printf("============ CANDIDATE GENERATION ============\n");
  printf("num_samples: %d\n", num_samples_);
  printf("num_threads: %d\n", num_threads_);
  printf("==============================================\n");

  Eigen::VectorXi cam_sources = Eigen::VectorXi::LinSpaced((360 / 3), 0, 360);
  all_cam_sources_.resize(cam_sources.size());
  Eigen::VectorXi::Map(&all_cam_sources_[0], cam_sources.rows()) = cam_sources;
}

void DataGenerator::generateData() {
  double total_time = 0.0;

  const candidate::HandGeometry &hand_geom =
      detector_->getHandSearchParameters().hand_geometry_;

  std::vector<std::string> objects = loadObjectNames(objects_file_location_);

  int store_step = 1;
  bool plot_grasps = false;
  int num_objects = objects.size();

  // debugging
  // num_objects = 1;
  // num_views_per_object_ = 20;

  std::vector<int> positives_list, negatives_list;
  std::vector<Instance> train_data, test_data;
  int num_train_views = num_views_per_object_ - test_views_.size();
  const int n_train = store_step * num_train_views * max_grasps_per_view_;
  const int n_test = store_step * test_views_.size() * max_grasps_per_view_;
  train_data.reserve(n_train);
  test_data.reserve(n_test);
  std::string train_file_path = output_root_ + "train.h5";
  std::string test_file_path = output_root_ + "test.h5";
  createDatasetsHDF5(train_file_path, objects.size() * n_train);
  createDatasetsHDF5(test_file_path, objects.size() * n_test);
  int train_offset = 0;
  int test_offset = 0;

  util::Plot plotter(1, 8);

  for (int i = 0; i < num_objects; i++) {
    printf("===> Generating images for object %d/%d: %s\n", i + 1, num_objects,
           objects[i].c_str());
    double t0 = omp_get_wtime();

    // Load mesh for ground truth.
    std::string prefix = data_root_ + objects[i];
    util::Cloud mesh = loadMesh(prefix + "_gt.pcd", prefix + "_gt_normals.csv");
    mesh.calculateNormalsOMP(num_threads_, normals_radius_);
    if (reverse_mesh_normals_) {
      mesh.setNormals(mesh.getNormals() * (-1.0));
    }

    for (int j = 0; j < num_views_per_object_; j++) {
      printf("===> Processing view %d/%d\n", j + 1, num_views_per_object_);

      std::vector<int> positives_view(0);
      std::vector<int> negatives_view(0);
      std::vector<std::unique_ptr<candidate::Hand>> labeled_grasps_view(0);
      std::vector<std::unique_ptr<cv::Mat>> images_view(0);

      // 1. Load point cloud.
      Eigen::Matrix3Xd view_points(3, 1);
      view_points << 0.0, 0.0, 0.0;  // TODO: Load camera position.
      util::Cloud cloud(prefix + "_" + std::to_string(j + 1) + ".pcd",
                        view_points);
      if (remove_nans_) {
        cloud.removeNans();
      }
      cloud.voxelizeCloud(voxel_size_views_);
      cloud.calculateNormalsOMP(num_threads_, normals_radius_);
      if (reverse_view_normals_) {
        cloud.setNormals(cloud.getNormals() * (-1.0));
      }

      while (positives_view.size() < min_grasps_per_view_) {
        cloud.subsampleUniformly(num_samples_);

        // 2. Find grasps in point cloud.
        std::vector<std::unique_ptr<candidate::Hand>> grasps;
        std::vector<std::unique_ptr<cv::Mat>> images;
        bool has_grasps = detector_->createGraspImages(cloud, grasps, images);

        if (plot_grasps) {
          // Plot plotter;
          //        plotter.plotNormals(cloud.getCloudOriginal(),
          //        cloud.getNormals());
          //        plotter.plotFingers(grasps, cloud.getCloudOriginal(),
          //        "Grasps on view");
          //        plotter.plotFingers3D(candidates,
          //        cloud_cam.getCloudOriginal(), "Grasps on view",
          //        hand_geom.outer_diameter_,
          //                                  hand_geom.finger_width_,
          //                                  hand_geom.depth_,
          //                                  hand_geom.height_);
        }

        // 3. Evaluate grasps against ground truth (mesh).
        printf("Eval GT ...\n");
        std::vector<int> labels = detector_->evalGroundTruth(mesh, grasps);

        // 4. Split grasps into positives and negatives.
        std::vector<int> positives;
        std::vector<int> negatives;
        splitInstances(labels, positives, negatives);

        if (positives_view.size() > 0 && positives.size() > 0) {
          for (int k = 0; k < positives.size(); k++) {
            positives[k] += images_view.size();
          }
        }
        if (negatives_view.size() > 0 && negatives.size() > 0) {
          for (int k = 0; k < negatives.size(); k++) {
            negatives[k] += images_view.size();
          }
        }

        images_view.insert(images_view.end(),
                           std::make_move_iterator(images.begin()),
                           std::make_move_iterator(images.end()));
        labeled_grasps_view.insert(labeled_grasps_view.end(),
                                   std::make_move_iterator(grasps.begin()),
                                   std::make_move_iterator(grasps.end()));

        positives_view.insert(positives_view.end(), positives.begin(),
                              positives.end());
        negatives_view.insert(negatives_view.end(), negatives.begin(),
                              negatives.end());
      }
      printf("positives, negatives found for this view: %zu, %zu\n",
             positives_view.size(), negatives_view.size());

      // 5. Balance the number of positives and negatives.
      balanceInstances(max_grasps_per_view_, positives_view, negatives_view,
                       positives_list, negatives_list);
      printf("#positives: %d, #negatives: %d\n", (int)positives_list.size(),
             (int)negatives_list.size());

      // 6. Assign instances to training or test data.
      if (std::find(test_views_.begin(), test_views_.end(), j) !=
          test_views_.end()) {
        addInstances(labeled_grasps_view, images_view, positives_list,
                     negatives_list, test_data);
        std::cout << "test view, # test data: " << test_data.size() << "\n";
      } else {
        addInstances(labeled_grasps_view, images_view, positives_list,
                     negatives_list, train_data);
        std::cout << "train view, # train data: " << train_data.size() << "\n";
      }
      printf("------------------------------------\n");
    }

    if ((i + 1) % store_step == 0) {
      // Shuffle the data.
      std::random_shuffle(train_data.begin(), train_data.end());
      std::random_shuffle(test_data.begin(), test_data.end());
      train_offset = insertIntoHDF5(train_file_path, train_data, train_offset);
      test_offset = insertIntoHDF5(test_file_path, test_data, test_offset);
      printf("train_offset: %d, test_offset: %d\n", train_offset, test_offset);
      train_data.clear();
      test_data.clear();
      train_data.reserve(store_step * num_views_per_object_ * 1000);
      test_data.reserve(store_step * num_views_per_object_ * 1000);
    }

    double ti = omp_get_wtime() - t0;
    total_time += ti;
    double num_objects_left = static_cast<double>(num_objects - i);
    double avg_time = total_time / (i + 1);
    printf("Number of objects left: %3.4f\n", num_objects_left);
    printf("Time for this object: %4.2fs. Total time: %3.2fs.\n", ti,
           total_time);
    printf("Average time per object: %4.2fs.\n", avg_time);
    printf(
        "Estimated time remaining (based on time for this object): %4.4fh or "
        "%4.4fs.\n",
        ti * num_objects_left * (1.0 / 3600.0), ti * num_objects_left);
    printf(
        "Estimated time remaining (based on average time per object): %4.4fh "
        "or %4.4fs.\n",
        avg_time * num_objects_left * (1.0 / 3600.0),
        avg_time * num_objects_left);
    printf("======================================\n\n");
  }

  // Store remaining data.
  if (train_data.size() > 0) {
    printf("Storing remaining instances ...\n");

    // Shuffle the data.
    std::random_shuffle(train_data.begin(), train_data.end());
    std::random_shuffle(test_data.begin(), test_data.end());
    train_offset = insertIntoHDF5(train_file_path, train_data, train_offset);
    test_offset = insertIntoHDF5(test_file_path, test_data, test_offset);
    printf("train_offset: %d, test_offset: %d\n", train_offset, test_offset);
  }

  printf("Generated %d training and test %d instances\n", train_offset,
         test_offset);
  std::string train_file_path_resized = output_root_ + "train2.h5";
  std::string test_file_path_resized = output_root_ + "test2.h5";
  printf("Resizing HDF5 training set to remove empty elements ...\n");
  reshapeHDF5(train_file_path, train_file_path_resized, "images", train_offset,
              chunk_size_, max_in_memory_);
  reshapeHDF5(train_file_path, train_file_path_resized, "labels", train_offset,
              chunk_size_, max_in_memory_);
  printf("Resizing HDF5 test set to remove empty elements ...\n");
  reshapeHDF5(test_file_path, test_file_path_resized, "images", test_offset,
              chunk_size_, max_in_memory_);
  reshapeHDF5(test_file_path, test_file_path_resized, "labels", test_offset,
              chunk_size_, max_in_memory_);
  printf("Wrote data to training and test databases\n");
}

void DataGenerator::createDatasetsHDF5(const std::string &filepath,
                                       int num_data) {
  printf("Opening HDF5 file at: %s\n", filepath.c_str());
  cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open(filepath);

  int n_dims_labels = 2;
  int dsdims_labels[n_dims_labels] = {num_data, 1};
  int chunks_labels[n_dims_labels] = {chunk_size_, dsdims_labels[1]};
  printf("Creating dataset <%s>: %d x %d\n", LABELS_DS_NAME.c_str(),
         dsdims_labels[0], dsdims_labels[1]);
  h5io->dscreate(n_dims_labels, dsdims_labels, CV_8UC1, LABELS_DS_NAME, 4,
                 chunks_labels);

  const descriptor::ImageGeometry &image_geom = detector_->getImageGeometry();
  int n_dims_images = 4;
  int dsdims_images[n_dims_images] = {
      num_data, image_geom.size_, image_geom.size_, image_geom.num_channels_};
  int chunks_images[n_dims_images] = {chunk_size_, dsdims_images[1],
                                      dsdims_images[2], dsdims_images[3]};
  printf("Creating dataset <%s>: %d x %d x %d x %d\n", IMAGES_DS_NAME.c_str(),
         dsdims_images[0], dsdims_images[1], dsdims_images[2],
         dsdims_images[3]);
  h5io->dscreate(n_dims_images, dsdims_images, CV_8UC1, IMAGES_DS_NAME,
                 n_dims_images, chunks_images);
  h5io->close();
}

void DataGenerator::reshapeHDF5(const std::string &in, const std::string &out,
                                const std::string &dataset, int num,
                                int chunk_size, int max_in_memory) {
  const int COMPRESSION_LEVEL = 9;

  cv::Ptr<cv::hdf::HDF5> h5io_in = cv::hdf::open(in);
  std::vector<int> dims = h5io_in->dsgetsize(dataset);
  std::vector<int> offsets(dims.size());
  for (int i = 0; i < dims.size(); i++) {
    offsets[i] = 0;
  }
  printf("Resizing dataset %s of length %d to %d\n", dataset.c_str(), dims[0],
         num);

  std::vector<int> chunks(dims.begin(), dims.end());
  chunks[0] = chunk_size;
  dims[0] = num;
  int type = h5io_in->dsgettype(dataset);
  cv::Ptr<cv::hdf::HDF5> h5io_out = cv::hdf::open(out);
  h5io_out->dscreate(dims.size(), &dims[0], type, dataset, COMPRESSION_LEVEL,
                     &chunks[0]);

  int steps = num / max_in_memory + 1;
  dims[0] = max_in_memory;
  for (int i = 0; i < steps; i++) {
    cv::Mat mat;
    dims[0] = (i == steps - 1) ? num - offsets[0] : max_in_memory;
    h5io_in->dsread(mat, dataset, offsets, dims);
    printf("Processing block %d/%d: %d to %d\n", i + 1, steps, offsets[0],
           offsets[0] + dims[0]);
    if (!mat.isContinuous()) {
      std::cout << "Error: matrix is not continuous! Consider resizing using "
                   "python.\n";
      break;
    }
    h5io_out->dsinsert(mat, dataset, offsets);
    offsets[0] += max_in_memory;
  }

  h5io_in->close();
  h5io_out->close();
}

util::Cloud DataGenerator::loadMesh(const std::string &mesh_file_path,
                                    const std::string &normals_file_path) {
  // Load mesh for ground truth.
  std::cout << " mesh_file_path: " << mesh_file_path << '\n';
  std::cout << " normals_file_path: " << normals_file_path << '\n';

  // Set the position from which the camera sees the point cloud.
  Eigen::Matrix3Xd view_points(3, 1);
  view_points << 0.0, 0.0, 0.0;

  // Load the point cloud.
  util::Cloud mesh_cloud_cam(mesh_file_path, view_points);

  // Load surface normals for the mesh.
  mesh_cloud_cam.setNormalsFromFile(normals_file_path);
  printf("Loaded mesh with %d points.\n",
         (int)mesh_cloud_cam.getCloudProcessed()->size());

  return mesh_cloud_cam;
}

std::vector<std::string> DataGenerator::loadObjectNames(
    const std::string &objects_file_location) {
  std::ifstream in;
  in.open(objects_file_location.c_str());
  std::string line;
  std::vector<std::string> objects;

  while (std::getline(in, line)) {
    std::stringstream lineStream(line);
    std::string object;
    std::getline(lineStream, object, '\n');
    std::cout << object << "\n";
    objects.push_back(object);
  }

  return objects;
}

void DataGenerator::splitInstances(const std::vector<int> &labels,
                                   std::vector<int> &positives,
                                   std::vector<int> &negatives) {
  positives.resize(0);
  negatives.resize(0);

  for (int i = 0; i < labels.size(); i++) {
    if (labels[i] == 1) {
      positives.push_back(i);
    } else {
      negatives.push_back(i);
    }
  }

  printf("#grasps: %zu, #positives: %zu, #negatives: %zu\n", labels.size(),
         positives.size(), negatives.size());
}

void DataGenerator::balanceInstances(int max_grasps_per_view,
                                     const std::vector<int> &positives_in,
                                     const std::vector<int> &negatives_in,
                                     std::vector<int> &positives_out,
                                     std::vector<int> &negatives_out) {
  size_t end;
  positives_out.resize(0);
  negatives_out.resize(0);

  if (negatives_in.size() > positives_in.size()) {
    end =
        std::min(positives_in.size(), (size_t)floor(0.5 * max_grasps_per_view));
    positives_out.insert(positives_out.end(), positives_in.begin(),
                         positives_in.begin() + end);
    negatives_out.insert(negatives_out.end(), negatives_in.begin(),
                         negatives_in.begin() + end);
  } else {
    end =
        std::min(negatives_in.size(), (size_t)floor(0.5 * max_grasps_per_view));
    negatives_out.insert(negatives_out.end(), negatives_in.begin(),
                         negatives_in.begin() + end);
    positives_out.insert(positives_out.end(), positives_in.begin(),
                         positives_in.begin() + end);
  }
}

void DataGenerator::addInstances(
    const std::vector<std::unique_ptr<candidate::Hand>> &grasps,
    std::vector<std::unique_ptr<cv::Mat>> &images,
    const std::vector<int> &positives, const std::vector<int> &negatives,
    std::vector<Instance> &dataset) {
  for (int k = 0; k < positives.size(); k++) {
    int idx = positives[k];
    if (!images[idx]) {
      printf(" => idx: %d is nullptr!\n", idx);
      char c;
      std::cin >> c;
    }
    dataset.push_back(
        Instance(std::move(images[idx]), grasps[idx]->isFullAntipodal()));
  }

  for (int k = 0; k < negatives.size(); k++) {
    int idx = negatives[k];
    if (!images[idx]) {
      printf(" => idx: %d is nullptr!\n", idx);
      char c;
      std::cin >> c;
    }
    dataset.push_back(
        Instance(std::move(images[idx]), grasps[idx]->isFullAntipodal()));
  }
}

int DataGenerator::insertIntoHDF5(const std::string &file_path,
                                  const std::vector<Instance> &dataset,
                                  int offset) {
  if (dataset.empty()) {
    printf("Error: Dataset is empty!\n");
    return offset;
  }

  printf("Storing %d items in HDF5: %s ... \n", (int)dataset.size(),
         file_path.c_str());

  cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open(file_path);
  printf("  Opened HDF5 file\n");

  const int num = static_cast<int>(dataset.size());
  const int rows = dataset[0].image_->rows;
  const int cols = dataset[0].image_->cols;
  const int channels = dataset[0].image_->channels();

  cv::Mat labels(dataset.size(), 1, CV_8UC1, cv::Scalar(0.0));

  for (int i = 0; i < dataset.size(); i++) {
    labels.at<uchar>(i) = (uchar)dataset[i].label_;
  }

  const int dims_images = 4;
  int dsdims_images[dims_images] = {num, rows, cols, channels};
  cv::Mat images(dims_images, dsdims_images, CV_8UC1, cv::Scalar(0.0));
  const int dims_image = 3;
  int dsdims_image[dims_image] = {rows, cols, channels};

  for (int i = 0; i < dataset.size(); i++) {
    if (!dataset[i].image_) {
      printf("FATAL ERROR! %d is nullptr\n", i);
      char c;
      std::cin >> c;
    }
    copyMatrix(*dataset[i].image_, images, i, dsdims_image);
  }

  printf("  Inserting into images dataset ...\n");
  const int dims_offset_images = 4;
  int offsets_images[dims_offset_images] = {offset, 0, 0, 0};
  h5io->dsinsert(images, IMAGES_DS_NAME, offsets_images);

  printf("  Inserting into labels dataset ...\n");
  const int dims_offset_labels = 2;
  int offsets_labels[dims_offset_labels] = {offset, 0};
  h5io->dsinsert(labels, LABELS_DS_NAME, offsets_labels);

  h5io->close();

  return offset + static_cast<int>(images.size[0]);
}

void DataGenerator::storeHDF5(const std::vector<Instance> &dataset,
                              const std::string &file_location) {
  const std::string IMAGE_DS_NAME = "images";
  const std::string LABELS_DS_NAME = "labels";

  if (dataset.empty()) {
    printf("Error: Dataset is empty!\n");
    return;
  }

  printf("Storing data as HDF5 at: %s\n", file_location.c_str());

  int n_dims = 3;
  int dsdims_images[n_dims] = {static_cast<int>(dataset.size()),
                               dataset[0].image_->rows,
                               dataset[0].image_->cols};
  cv::Mat images(n_dims, dsdims_images, CV_8UC(dataset[0].image_->channels()),
                 cv::Scalar(0.0));
  cv::Mat labels(dataset.size(), 1, CV_8UC1, cv::Scalar(0.0));
  printf(" Dataset dimensions: %d x %d x %d x%d\n", dsdims_images[0],
         dsdims_images[1], dsdims_images[2], images.channels());
  int dims_image[n_dims] = {dataset[0].image_->rows, dataset[0].image_->cols,
                            dataset[0].image_->channels()};

  for (int i = 0; i < dataset.size(); i++) {
    // ranges don't seem to work
    //    std::vector<cv::Range> ranges;
    //    ranges.push_back(cv::Range(i,i+1));
    //    ranges.push_back(cv::Range(0,60));
    //    ranges.push_back(cv::Range(0,60));
    //    data(&ranges[0]) = dataset[i].image_.clone();
    //    dataset[0].image_->copyTo(data(&ranges[0]));
    copyMatrix(*dataset[i].image_, images, i, dims_image);
    labels.at<uchar>(i) = (uchar)dataset[i].label_;
    //    printf("dataset.label: %d,  labels(i): %d\n", dataset[i].label_,
    //    labels.at<uchar>(i));
  }

  cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open(file_location);
  if (!h5io->hlexists(IMAGE_DS_NAME)) {
    int n_dims_labels = 2;
    int dsdims_labels[n_dims_labels] = {static_cast<int>(dataset.size()), 1};
    printf("Creating dataset <labels> ...\n");
    h5io->dscreate(n_dims_labels, dsdims_labels, CV_8UC1, LABELS_DS_NAME, 9);
    printf("Writing dataset <labels> ...\n");
    h5io->dswrite(labels, LABELS_DS_NAME);
    printf("Wrote dataset of length %d to HDF5.\n", (int)dataset.size());

    printf("Creating dataset <images> ...\n");
    int chunks[n_dims] = {chunk_size_, dsdims_images[1], dsdims_images[2]};
    h5io->dscreate(n_dims, dsdims_images, CV_8UC(dataset[0].image_->channels()),
                   IMAGE_DS_NAME, 9, chunks);
    printf("Writing dataset <images> ...\n");
    h5io->dswrite(images, IMAGE_DS_NAME);
  }
  h5io->close();
}

void DataGenerator::printMatrix(const cv::Mat &mat) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 60; j++) {
      for (int k = 0; k < 60; k++) {
        for (int l = 0; l < 15; l++) {
          int idx[4] = {i, j, k, l};
          if (mat.at<uchar>(idx) > 0)
            printf("%d,%d,%d,%d: %d\n", i, j, k, l, (int)mat.at<uchar>(idx));
          //          std::cout << i << ", " << j << ", " << k << ", l" << ": "
          //          << (int) mat.at<uchar>(idx) << " \n";
        }
      }
    }
  }
}

void DataGenerator::printMatrix15(const cv::Mat &mat) {
  for (int j = 0; j < 60; j++) {
    for (int k = 0; k < 60; k++) {
      for (int l = 0; l < 15; l++) {
        int idx[3] = {j, k, l};
        if (mat.at<uchar>(idx) > 0)
          printf("%d,%d,%d: %d\n", j, k, l, (int)mat.at<uchar>(idx));
      }
    }
  }
}

// src: multi-channels image; dst: multi-dimensional matrix
void DataGenerator::copyMatrix(const cv::Mat &src, cv::Mat &dst, int idx_in,
                               int *dims_img) {
  const int rows = dims_img[0];
  const int cols = dims_img[1];
  const int channels = dims_img[2];
  for (int j = 0; j < rows; j++) {
    for (int k = 0; k < cols; k++) {
      for (int l = 0; l < channels; l++) {
        int idx_dst[4] = {idx_in, j, k, l};
        dst.at<uchar>(idx_dst) = src.ptr<uchar>(j)[k * channels + l];
      }
    }
  }
}

util::Cloud DataGenerator::createMultiViewCloud(const std::string &object,
                                                int camera,
                                                const std::vector<int> angles,
                                                int reference_camera) const {
  const std::string prefix = data_root_ + object;
  const std::string np_str = "NP" + std::to_string(camera) + "_";
  const std::string cloud_prefix = prefix + "/clouds/" + np_str;
  const std::string pose_prefix = prefix + "/poses/" + np_str;

  PointCloudRGB::Ptr concatenated_cloud(new PointCloudRGB);
  Eigen::Matrix3Xf camera_positions(3, angles.size());
  std::vector<int> sizes;
  sizes.resize(angles.size());
  util::Cloud tmp;

  for (int i = 0; i < angles.size(); i++) {
    std::string cloud_filename =
        cloud_prefix + std::to_string(angles[i]) + ".pcd";
    printf("cloud_filename: %s\n", cloud_filename.c_str());
    PointCloudRGB::Ptr cloud = tmp.loadPointCloudFromFile(cloud_filename);

    Eigen::Matrix4f T =
        calculateTransform(object, camera, angles[i], reference_camera);
    pcl::transformPointCloud(*cloud, *cloud, T);

    pcl::io::savePCDFileASCII(
        object + np_str + std::to_string(angles[i]) + ".pcd", *cloud);

    *concatenated_cloud += *cloud;
    camera_positions.col(i) = T.col(3).head(3);
    sizes[i] = cloud->size();
  }

  Eigen::MatrixXi camera_sources =
      Eigen::MatrixXi::Zero(angles.size(), concatenated_cloud->size());
  int start = 0;
  for (int i = 0; i < angles.size(); i++) {
    camera_sources.block(i, start, 1, sizes[i]).setConstant(1);
    start += sizes[i];
  }

  util::Cloud multi_view_cloud(concatenated_cloud, camera_sources,
                               camera_positions.cast<double>());

  pcl::io::savePCDFileASCII(object + np_str + "_merged.pcd",
                            *multi_view_cloud.getCloudOriginal());

  return multi_view_cloud;
}

Eigen::Matrix4f DataGenerator::calculateTransform(const std::string &object,
                                                  int camera, int angle,
                                                  int reference_camera) const {
  std::string pose_filename = data_root_ + object + "/poses/NP" +
                              std::to_string(reference_camera) + "_" +
                              std::to_string(angle) + "_pose.h5";
  std::string table_from_ref_key = "H_table_from_reference_camera";
  printf("pose_filename: %s\n", pose_filename.c_str());
  Eigen::Matrix4f T_table_from_ref =
      readPoseFromHDF5(pose_filename, table_from_ref_key);

  std::string calibration_filename = data_root_ + object + "/calibration.h5";
  std::string cam_from_ref_key = "H_NP" + std::to_string(camera) + "_from_NP" +
                                 std::to_string(reference_camera);
  printf("calibration_filename: %s\n", calibration_filename.c_str());
  printf("cam_from_ref_key: %s\n", cam_from_ref_key.c_str());
  Eigen::Matrix4f T_cam_from_ref =
      readPoseFromHDF5(calibration_filename, cam_from_ref_key);

  Eigen::Matrix4f T = T_table_from_ref * T_cam_from_ref.inverse();

  return T;
}

Eigen::Matrix4f DataGenerator::readPoseFromHDF5(
    const std::string &hdf5_filename, const std::string &dsname) const {
  cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open(hdf5_filename);
  cv::Mat mat_cv(4, 4, CV_32FC1);
  h5io->dsread(mat_cv, dsname);
  Eigen::Matrix4f mat_eigen;
  cv::cv2eigen(mat_cv, mat_eigen);
  h5io->close();

  return mat_eigen;
}

}  // namespace gpd
