#include <gpd/net/caffe_classifier.h>

namespace gpd {
namespace net {

CaffeClassifier::CaffeClassifier(const std::string &model_file,
                                 const std::string &weights_file,
                                 Classifier::Device device, int batch_size) {
  // Initialize Caffe.
  switch (device) {
    case Classifier::Device::eGPU:
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
      break;
    default:
      caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  // Load pretrained network.
  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  input_layer_ = boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(
      net_->layer_by_name("data"));
  input_layer_->set_batch_size(batch_size);
}

std::vector<float> CaffeClassifier::classifyImages(
    const std::vector<std::unique_ptr<cv::Mat>> &image_list) {
  int batch_size = input_layer_->batch_size();
  int num_iterations = (int)ceil(image_list.size() / (double)batch_size);
  float loss = 0.0;
  std::cout << "# images: " << image_list.size()
            << ", # iterations: " << num_iterations
            << ", batch size: " << batch_size << "\n";

  std::vector<float> predictions;

  // Process the images in batches.
  for (int i = 0; i < num_iterations; i++) {
    std::vector<cv::Mat> sub_image_list;

    if (i < num_iterations - 1) {
      for (int j = 0; j < batch_size; j++) {
        sub_image_list.push_back(*image_list[i * batch_size + j]);
      }
    } else {
      for (int j = 0; j < image_list.size() - i * batch_size; j++) {
        sub_image_list.push_back(*image_list[i * batch_size + j]);
      }
      for (int j = sub_image_list.size(); j < batch_size; j++) {
        cv::Mat empty_mat(input_layer_->height(), input_layer_->width(),
                          CV_8UC(input_layer_->channels()), cv::Scalar(0.0));
        sub_image_list.push_back(empty_mat);
      }
    }

    std::vector<int> label_list;
    label_list.resize(sub_image_list.size());

    for (int j = 0; j < label_list.size(); j++) {
      label_list[j] = 0;
    }

    // Classify the batch.
    input_layer_->AddMatVector(sub_image_list, label_list);
    std::vector<caffe::Blob<float> *> results = net_->Forward(&loss);
    std::vector<float> out(results[0]->cpu_data(),
                           results[0]->cpu_data() + results[0]->count());

    for (int l = 0; l < results[0]->count() / results[0]->channels(); l++) {
      predictions.push_back(out[2 * l + 1] - out[2 * l]);
    }
  }

  return predictions;
}

}  // namespace net
}  // namespace gpd
