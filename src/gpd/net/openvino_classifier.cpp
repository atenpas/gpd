#include <gpd/net/openvino_classifier.h>

namespace gpd {
namespace net {

using namespace InferenceEngine;

std::map<Classifier::Device, InferenceEngine::TargetDevice>
    OpenVinoClassifier::device_map_ = {
        {Classifier::Device::eCPU, TargetDevice::eCPU},
        {Classifier::Device::eGPU, TargetDevice::eGPU},
        {Classifier::Device::eVPU, TargetDevice::eMYRIAD},
        {Classifier::Device::eFPGA, TargetDevice::eFPGA}};

OpenVinoClassifier::OpenVinoClassifier(const std::string &model_file,
                                       const std::string &weights_file,
                                       Classifier::Device device,
                                       int batch_size) {
  InferenceEngine::PluginDispatcher dispatcher({"../../../lib/intel64", ""});
  plugin_ = InferencePlugin(dispatcher.getSuitablePlugin(device_map_[device]));

  CNNNetReader network_reader;
  network_reader.ReadNetwork(model_file);
  network_reader.ReadWeights(weights_file);
  network_ = network_reader.getNetwork();
  network_.setBatchSize(batch_size);

  InputsDataMap input = network_.getInputsInfo();
  input.begin()->second->setPrecision(Precision::FP32);

  OutputsDataMap output = network_.getOutputsInfo();
  output.begin()->second->setPrecision(Precision::FP32);

  infer_request_ = plugin_.LoadNetwork(network_, {}).CreateInferRequest();
  auto output_name = output.begin()->first;
  output_blob_ = infer_request_.GetBlob(output_name);
}

std::vector<float> OpenVinoClassifier::classifyImages(
    const std::vector<std::unique_ptr<cv::Mat>> &image_list) {
  std::vector<float> predictions(0);
  InputsDataMap input_info = network_.getInputsInfo();

  for (const auto &item : input_info) {
    Blob::Ptr input = infer_request_.GetBlob(item.first);
    SizeVector dims = input->getTensorDesc().getDims();
    size_t channels = dims[1];
    size_t rows = dims[2];
    size_t cols = dims[3];
    size_t image_size = rows * cols;
    auto data =
        input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    int num_iter = (int)ceil(image_list.size() / (double)getBatchSize());

    for (size_t i = 0; i < num_iter; i++) {
      int n = std::min(getBatchSize(),
                       (int)(image_list.size() - i * getBatchSize()));
      for (size_t b = 0; b < n; b++) {
        int image_id = i * getBatchSize() + b;
        for (int r = 0; r < rows; r++) {
          for (int c = 0; c < cols; c++) {
            for (int ch = 0; ch < channels; ch++) {
              int src_idx[3] = {r, c, ch};
              int dst_idx =
                  b * image_size * channels + ch * image_size + r * cols + c;
              data[dst_idx] = image_list[image_id]->at<uchar>(src_idx);
            }
          }
        }
        if (n < getBatchSize()) {
          for (int b = n; b < getBatchSize(); b++) {
            for (int r = 0; r < rows; r++) {
              for (int c = 0; c < cols; c++) {
                for (int ch = 0; ch < channels; ch++) {
                  int dst_idx = b * image_size * channels + ch * image_size +
                                r * cols + c;
                  data[dst_idx] = 0;
                }
              }
            }
          }
        }
      }
      infer_request_.Infer();

      auto output_data =
          output_blob_->buffer()
              .as<PrecisionTrait<Precision::FP32>::value_type *>();
      const int resultsCnt = output_blob_->size() / getBatchSize();

      for (int j = 0; j < n; j++) {
        predictions.push_back(output_data[2 * j + 1] - output_data[2 * j]);
      }
    }
  }

  return predictions;
}

int OpenVinoClassifier::getBatchSize() const { return network_.getBatchSize(); }

}  // namespace net
}  // namespace gpd
