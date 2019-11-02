#include <gpd/descriptor/image_geometry.h>

namespace gpd {
namespace descriptor {

ImageGeometry::ImageGeometry()
    : outer_diameter_(0.0),
      depth_(0.0),
      height_(0.0),
      size_(0),
      num_channels_(0) {}

ImageGeometry::ImageGeometry(double outer_diameter, double depth, double height,
                             int size, int num_channels)
    : outer_diameter_(outer_diameter),
      depth_(depth),
      height_(height),
      size_(size),
      num_channels_(num_channels) {}

ImageGeometry::ImageGeometry(const std::string &filepath) {
  util::ConfigFile config_file(filepath);
  config_file.ExtractKeys();
  outer_diameter_ = config_file.getValueOfKey<double>("volume_width", 0.10);
  depth_ = config_file.getValueOfKey<double>("volume_depth", 0.06);
  height_ = config_file.getValueOfKey<double>("volume_height", 0.02);
  size_ = config_file.getValueOfKey<int>("image_size", 60);
  num_channels_ = config_file.getValueOfKey<int>("image_num_channels", 15);
}

std::ostream &operator<<(std::ostream &stream,
                         const ImageGeometry &image_geometry) {
  stream << "============ GRASP IMAGE GEOMETRY ===============\n";
  stream << "volume width: " << image_geometry.outer_diameter_ << "\n";
  stream << "volume depth: " << image_geometry.depth_ << "\n";
  stream << "volume height: " << image_geometry.height_ << "\n";
  stream << "image_size: " << image_geometry.size_ << "\n";
  stream << "image_num_channels: " << image_geometry.num_channels_ << "\n";
  stream << "=================================================\n";

  return stream;
}

}  // namespace descriptor
}  // namespace gpd
