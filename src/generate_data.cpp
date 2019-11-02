#include <gpd/data_generator.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace gpd {
namespace apps {
namespace generate_data {

int DoMain(int argc, char **argv) {
  // Read arguments from command line.
  if (argc < 2) {
    std::cout << "Error: Not enough input arguments!\n\n";
    std::cout << "Usage: generate_data CONFIG_FILE\n\n";
    std::cout << "Generate data using parameters from CONFIG_FILE (*.cfg).\n\n";
    return (-1);
  }

  // Seed the random number generator.
  std::srand(std::time(0));

  // Read path to config file.
  std::string config_filename = argv[1];

  // Create training data.
  DataGenerator generator(config_filename);

  generator.generateData();

  // std::vector<int> idx;
  // for (int i = 0; i < 120; i++) {
  //   idx.push_back(i * 3);
  // }
  //  idx.push_back(1);
  //  idx.push_back(5);
  //  idx.push_back(15);
  //  idx.push_back(25);
  //  idx.push_back(31);
  // Cloud cloud = generator.createMultiViewCloud("blue_clover_baby_toy", idx);
  // int camera = 1;
  // int reference_camera = 5;
  // util::Cloud cloud = generator.createMultiViewCloud(
  //     "pop_tarts_strawberry", camera, idx, reference_camera);
  // pcl::io::savePCDFileASCII("test_pcd.pcd", *cloud.getCloudOriginal());

  //  cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open("test.hdf5");
  //  std::string IMAGE_DS_NAME = "images";
  //  if (!h5io->hlexists(IMAGE_DS_NAME))
  //  {
  //    int n_dims = 3;
  //    int dsdims[n_dims] = { 505000, 60, 60 };
  //    int chunks[n_dims] = {  10000, 60, 60 };
  //    cv::Mat images(n_dims, dsdims, CV_8UC(15), cv::Scalar(0.0));
  //
  //    printf("Creating dataset <images> ...\n");
  //    h5io->dscreate(n_dims, dsdims, CV_8UC(15), IMAGE_DS_NAME, 9, chunks);
  //    printf("Writing dataset <images> ...\n");
  //    h5io->dswrite(images, IMAGE_DS_NAME);
  //  }
  //  h5io->close();

  return 0;
}

}  // namespace generate_data
}  // namespace apps
}  // namespace gpd

int main(int argc, char *argv[]) {
  return gpd::apps::generate_data::DoMain(argc, argv);
}
