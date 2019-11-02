#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/hdf5.hpp>
using namespace cv;

static void write_root_group_single_channel() {
  std::cout << "write_root_group_single_channel\n";
  String filename = "root_group_single_channel.h5";
  String dataset_name =
      "/single";  // Note that it is a child of the root group /
  // prepare data
  Mat data;
  data = (cv::Mat_<float>(2, 3) << 0, 1, 2, 3, 4, 5, 6);
  Ptr<hdf::HDF5> h5io = hdf::open(filename);
  // write data to the given dataset
  // the dataset "/single" is created automatically, since it is a child of the
  // root
  h5io->dswrite(data, dataset_name);
  Mat expected;
  h5io->dsread(expected, dataset_name);
  double diff = norm(data - expected);
  CV_Assert(abs(diff) < 1e-10);
  h5io->close();
}

static void write_single_channel() {
  String filename = "single_channel.h5";
  String parent_name = "/data";
  String dataset_name = parent_name + "/single";
  // prepare data
  Mat data;
  data = (cv::Mat_<float>(2, 3) << 0, 1, 2, 3, 4, 5);
  Ptr<hdf::HDF5> h5io = hdf::open(filename);
  // first we need to create the parent group
  if (!h5io->hlexists(parent_name)) h5io->grcreate(parent_name);
  // create the dataset if it not exists
  if (!h5io->hlexists(dataset_name))
    h5io->dscreate(data.rows, data.cols, data.type(), dataset_name);
  // the following is the same with the above function
  // write_root_group_single_channel()
  h5io->dswrite(data, dataset_name);
  Mat expected;
  h5io->dsread(expected, dataset_name);

  std::cout << "expected:\n" << expected << "\n";

  double diff = norm(data - expected);
  CV_Assert(abs(diff) < 1e-10);
  h5io->close();
}

/*
 * creating, reading and writing multiple-channel matrices
 * are the same with single channel matrices
 */
static void write_multiple_channels() {
  String filename = "two_channels.h5";
  String parent_name = "/data";
  String dataset_name = parent_name + "/two_channels";
  // prepare data
  Mat data(2, 3, CV_32SC2);
  for (size_t i = 0; i < data.total() * data.channels(); i++)
    ((int *)data.data)[i] = (int)i;
  Ptr<hdf::HDF5> h5io = hdf::open(filename);
  // first we need to create the parent group
  if (!h5io->hlexists(parent_name)) h5io->grcreate(parent_name);
  // create the dataset if it not exists
  if (!h5io->hlexists(dataset_name))
    h5io->dscreate(data.rows, data.cols, data.type(), dataset_name);
  // the following is the same with the above function
  // write_root_group_single_channel()
  h5io->dswrite(data, dataset_name);
  Mat expected;
  h5io->dsread(expected, dataset_name);
  double diff = norm(data - expected);
  CV_Assert(abs(diff) < 1e-10);
  h5io->close();
}

int main(int argc, char *argv[]) {
  write_root_group_single_channel();
  write_single_channel();
  write_multiple_channels();

  Ptr<hdf::HDF5> h5io = hdf::open(argv[1]);
  Mat expected;
  h5io->dsread(expected, "H_table_from_reference_camera");
  std::cout << "H_table_from_reference_camera:\n" << expected << "\n";
  h5io->close();

  return 0;
}
