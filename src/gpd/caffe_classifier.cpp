#include "../../include/gpd/caffe_classifier.h"


CaffeClassifier::CaffeClassifier(const std::string& model_file, const std::string& weights_file, Classifier::Device device)
{
  // Initialize Caffe.
  switch (device)
  {
  case Classifier::Device::eGPU:
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    break;
  default:
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  // Load pretrained network.
  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  input_layer_ = boost::static_pointer_cast < caffe::MemoryDataLayer < float > >(net_->layer_by_name("data"));
}


std::vector<float> CaffeClassifier::classifyImages(const std::vector<cv::Mat>& image_list)
{
  int batch_size = input_layer_->batch_size();
  int num_iterations = (int) ceil(image_list.size() / (double) batch_size);
  float loss = 0.0;
  std::cout << "# images: " << image_list.size() << ", # iterations: " << num_iterations << ", batch size: "
    << batch_size << "\n";

  std::vector<float> predictions;

  // Process the images in batches.
  for (int i = 0; i < num_iterations; i++)
  {
    std::vector<cv::Mat>::const_iterator end_it;
    std::vector<cv::Mat> sub_image_list;

    if (i < num_iterations - 1)
    {
      end_it = image_list.begin() + (i + 1) * batch_size;
      sub_image_list.assign(image_list.begin() + i * batch_size, end_it);
    }
    // Fill the batch with empty images to match the required batch size.
    else
    {
      end_it = image_list.end();
      sub_image_list.assign(image_list.begin() + i * batch_size, end_it);
      std::cout << "Adding " << batch_size - sub_image_list.size() << " empty images to batch to match batch size.\n";

      for (int t = sub_image_list.size(); t < batch_size; t++)
      {
        cv::Mat empty_mat(input_layer_->height(), input_layer_->width(), CV_8UC(input_layer_->channels()), cv::Scalar(0.0));
        sub_image_list.push_back(empty_mat);
      }
    }

    std::vector<int> label_list;
    label_list.resize(sub_image_list.size());

    for (int j = 0; j < label_list.size(); j++)
    {
      label_list[j] = 0;
    }

    // Classify the batch.
    input_layer_->AddMatVector(sub_image_list, label_list);
    std::vector<caffe::Blob<float>*> results = net_->Forward(&loss);
    std::vector<float> out(results[0]->cpu_data(), results[0]->cpu_data() + results[0]->count());
//    std::cout << "#results: " << results.size() << ", " << results[0]->count() << "\n";

    for (int l = 0; l < results[0]->count() / results[0]-> channels(); l++)
    {
      predictions.push_back(out[2 * l + 1] - out[2 * l]);
//      std::cout << "positive score: " << out[2 * l + 1] << ", negative score: " << out[2 * l] << "\n";
    }
  }

  return predictions;
}
