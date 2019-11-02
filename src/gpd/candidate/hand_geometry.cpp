#include <gpd/candidate/hand_geometry.h>

namespace gpd {
namespace candidate {

HandGeometry::HandGeometry()
    : finger_width_(0.0),
      outer_diameter_(0.0),
      depth_(0.0),
      height_(0.0),
      init_bite_(0.0) {}

HandGeometry::HandGeometry(double finger_width, double outer_diameter,
                           double hand_depth, double hand_height,
                           double init_bite)
    : finger_width_(finger_width),
      outer_diameter_(outer_diameter),
      depth_(hand_depth),
      height_(hand_height),
      init_bite_(init_bite) {}

HandGeometry::HandGeometry(const std::string &filepath) {
  util::ConfigFile config_file(filepath);
  config_file.ExtractKeys();
  finger_width_ = config_file.getValueOfKey<double>("finger_width", 0.01);
  outer_diameter_ =
      config_file.getValueOfKey<double>("hand_outer_diameter", 0.12);
  depth_ = config_file.getValueOfKey<double>("hand_depth", 0.06);
  height_ = config_file.getValueOfKey<double>("hand_height", 0.02);
  init_bite_ = config_file.getValueOfKey<double>("init_bite", 0.01);
}

std::ostream &operator<<(std::ostream &stream,
                         const HandGeometry &hand_geometry) {
  stream << "============ HAND GEOMETRY ======================\n";
  stream << "finger_width: " << hand_geometry.finger_width_ << "\n";
  stream << "hand_outer_diameter: " << hand_geometry.outer_diameter_ << "\n";
  stream << "hand_depth: " << hand_geometry.depth_ << "\n";
  stream << "hand_height: " << hand_geometry.height_ << "\n";
  stream << "init_bite: " << hand_geometry.init_bite_ << "\n";
  stream << "=================================================\n";
  return stream;
}

}  // namespace candidate
}  // namespace gpd
