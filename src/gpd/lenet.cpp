#include <gpd/lenet.h>


Lenet::Lenet(int num_threads, const std::string& params_dir) : num_threads_(num_threads)
{
  double start = omp_get_wtime();

  int image_size = 60;
  int num_channels = 15;

  int num_filters = 20;
  int spatial_extent = 5;
  int stride = 1;
  int padding = 0;

  // Construct the network.
  conv1 = new ConvLayer(image_size, image_size, num_channels, num_filters, spatial_extent, stride, padding);
  conv2 = new ConvLayer(28, 28, 20, 50, 5, 1, 0);
  dense1 = new DenseLayer(500);
  dense2 = new DenseLayer(2);

  // Set weights and biases.
  std::vector<float> w_vec = readBinaryFileIntoVector(params_dir + "conv1_0.bin");
  std::vector<float> b_vec = readBinaryFileIntoVector(params_dir + "conv1_1.bin");
  conv1->setWeightsAndBiases(w_vec, b_vec);

  w_vec = readBinaryFileIntoVector(params_dir + "conv2_0.bin");
  b_vec = readBinaryFileIntoVector(params_dir + "conv2_1.bin");
  conv2->setWeightsAndBiases(w_vec, b_vec);

  std::vector<float> w_dense1 = readBinaryFileIntoVector(params_dir + "dense1_0.bin");
  std::vector<float> b_dense1 = readBinaryFileIntoVector(params_dir + "dense1_1.bin");
  dense1->setWeightsAndBiases(w_dense1, b_dense1);

  std::vector<float> w_dense2 = readBinaryFileIntoVector(params_dir + "dense2_0.bin");
  std::vector<float> b_dense2 = readBinaryFileIntoVector(params_dir + "dense2_1.bin");

  dense2->setWeightsAndBiases(w_dense2, b_dense2);

  x_conv2.resize(20*28*28);
  x_dense1.resize(50*12*12);
  x_dense2.resize(500);

  std::cout << "NET SETUP runtime: " << omp_get_wtime() - start << std::endl;
}


std::vector<float> Lenet::classifyImages(const std::vector<cv::Mat>& image_list)
{
  std::vector<float> y;
  y.resize(image_list.size());

#ifdef _OPENMP // parallelization using OpenMP
#pragma omp parallel for num_threads(num_threads_)
#endif
  for (int i = 0; i < image_list.size(); i++)
  {
    if (image_list[i].isContinuous())
    {
      std::vector<float> x = imageToArray(image_list[i]);

      std::vector<float> yi = forward(x);
//      std::cout << i << " -- positive score: " << yi[1] << ", negative score: " << yi[0] << "\n";
      y[i] = yi[1] - yi[0];
    }
  }

  return y;
}


std::vector<float> Lenet::forward(const std::vector<float>& x)
{
//  double start = omp_get_wtime();

  // 1st conv layer
  Eigen::MatrixXf H1 = conv1->forward(x);

  // 1st max pool layer
  Eigen::MatrixXf P1 = poolForward(H1, 2, 2);

  // 2nd conv layer
  double conv2_start = omp_get_wtime();
  Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> P1r(P1);
//  std::vector<float> x_conv2;
  x_conv2.assign(P1r.data(), P1r.data() + P1r.size());
  Eigen::MatrixXf H2 = conv2->forward(x_conv2);
//  std::cout << "CONV2 runtime: " << omp_get_wtime() - conv2_start << std::endl;

  // 2nd max pool layer
  Eigen::MatrixXf P2 = poolForward(H2, 2, 2);

  // Flatten the output of the 2nd max pool layer.
  Eigen::Map<Eigen::VectorXf> f1(P2.data(), P2.size());

  // 1st inner product layer
//  double dense1_start = omp_get_wtime();
  Eigen::VectorXf::Map(&x_dense1[0], f1.size()) = f1;
  Eigen::MatrixXf H3 = dense1->forward(x_dense1);
//  std::cout << "DENSE1 runtime: " << omp_get_wtime() - dense1_start << std::endl;

  // RELU layer
  H3 = H3.cwiseMax(0);

  // 2nd inner product layer (output layer)
  Eigen::Map<Eigen::VectorXf> f2(H3.data(), H3.size());
  Eigen::VectorXf::Map(&x_dense2[0], f2.size()) = f2;
  Eigen::MatrixXf Y = dense2->forward(x_dense2);

//  std::cout << "FORWARD PASS runtime: " << omp_get_wtime() - start << std::endl;
//  int n = Eigen::nbThreads();
//  std::cout << "# CPU threads: " << n << std::endl;

  std::vector<float> y;
  y.assign(Y.data(), Y.data() + Y.size());
  return y;
}


Eigen::MatrixXf Lenet::poolForward(const Eigen::MatrixXf& X, int filter_size, int stride) const
{
  int depth = X.rows();
  int width_in = sqrt(X.cols());
  int width_out = (width_in - filter_size)/stride + 1;

  int block_size = filter_size*filter_size;

  Eigen::MatrixXf M(depth, width_out*width_out);

  for (int i = 0; i < X.rows(); i++)
  {
    int row_in = 0;
    int col_in = 0;

    for (int j = 0; j < width_out*width_out; j++)
    {
      int start = row_in*width_in + col_in;

      Eigen::VectorXf block(filter_size*filter_size);
      block <<  X(i, start),              X(i, start + 1),
                X(i, start + width_in),   X(i, start + width_in + 1);
      M(i,j) = block.maxCoeff();

      col_in += stride;
      if (col_in == width_in)
      {
        row_in += stride;
        col_in = 0;
      }
    }
  }

  return M;
}


std::vector<float> Lenet::readFileLineByLineIntoVector(const std::string& location)
{
  std::ifstream file;
  std::string line;
  file.open(location.c_str());
  std::vector<float> vals;
  while (std::getline (file, line))
  {
    vals.push_back(atof(line.c_str()));
  }

  file.close();

  return vals;
}


std::vector<float> Lenet::readBinaryFileIntoVector(const std::string& location)
{
  std::vector<float> vals;

  std::ifstream file(location.c_str(), std::ios::binary|std::ios::in);
  if (!file.is_open())
  {
    std::cout << "Error: Cannot open file!\n";
    vals.resize(0);
    return vals;
  }

  float x;
  while (file.read(reinterpret_cast<char*>(&x), sizeof(float)))
  {
    vals.push_back(x);
  }

  file.close();

  return vals;
}


std::vector<float> Lenet::imageToArray(const cv::Mat& img) const
{
  std::vector<float> x;
  x.resize(img.channels()*img.rows*img.cols);
  int k = 0;

  for (int channel = 0; channel < img.channels(); channel++)
  {
    for (int row = 0; row < img.rows; row++)
    {
      const uchar* ptr = img.ptr(row);

      for (int col = 0; col < img.cols; col++)
      {
          const uchar* uc_pixel = ptr;
          x[k] = uc_pixel[channel];
          ptr += 15;
          k++;
      }
    }
  }

  return x;
}
