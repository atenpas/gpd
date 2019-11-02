#include <gpd/candidate/hand.h>

namespace gpd {
namespace candidate {

Hand::Hand() : grasp_width_(0.0), label_(0.0, false, false) {}

Hand::Hand(const Eigen::Vector3d &sample, const Eigen::Matrix3d &frame,
           const FingerHand &finger_hand, double grasp_width)
    : sample_(sample), grasp_width_(grasp_width), label_(0.0, false, false) {
  orientation_ = frame;

  construct(finger_hand);
}

Hand::Hand(const Eigen::Vector3d &sample, const Eigen::Matrix3d &frame,
           const FingerHand &finger_hand)
    : sample_(sample), grasp_width_(0.0), label_(0.0, false, false) {
  orientation_ = frame;

  construct(finger_hand);
}

void Hand::construct(const FingerHand &finger_hand) {
  closing_box_.top_ = finger_hand.getTop();
  closing_box_.bottom_ = finger_hand.getBottom();
  closing_box_.center_ = finger_hand.getCenter();

  calculateGraspPositions(finger_hand);

  // Determine the index of the finger placement that resulted in this grasp.
  const Eigen::Array<bool, 1, Eigen::Dynamic> &indices = finger_hand.getHand();
  for (int i = 0; i < indices.size(); i++) {
    if (indices[i] == true) {
      finger_placement_index_ = i;
      break;
    }
  }
}

void Hand::calculateGraspPositions(const FingerHand &finger_hand) {
  Eigen::Vector3d pos_bottom;
  pos_bottom << getBottom(), finger_hand.getCenter(), 0.0;
  position_ = getFrame() * pos_bottom + sample_;
}

void Hand::writeHandsToFile(const std::string &filename,
                            const std::vector<Hand> &hands) const {
  std::ofstream myfile;
  myfile.open(filename.c_str());

  for (int i = 0; i < hands.size(); i++) {
    std::cout << "Hand " << i << std::endl;
    print();

    myfile << vectorToString(hands[i].getPosition())
           << vectorToString(hands[i].getAxis())
           << vectorToString(hands[i].getApproach())
           << vectorToString(hands[i].getBinormal())
           << std::to_string(hands[i].getGraspWidth()) << "\n";
  }

  myfile.close();
}

void Hand::print() const {
  std::cout << "position: " << getPosition().transpose() << std::endl;
  std::cout << "approach: " << getApproach().transpose() << std::endl;
  std::cout << "binormal: " << getBinormal().transpose() << std::endl;
  std::cout << "axis: " << getAxis().transpose() << std::endl;
  std::cout << "score: " << getScore() << std::endl;
  std::cout << "full-antipodal: " << isFullAntipodal() << std::endl;
  std::cout << "half-antipodal: " << isHalfAntipodal() << std::endl;
  std::cout << "closing box:\n";
  std::cout << " bottom: " << getBottom() << std::endl;
  std::cout << " top: " << getTop() << std::endl;
  std::cout << " center: " << getTop() << std::endl;
}

std::string Hand::vectorToString(const Eigen::VectorXd &v) const {
  std::string s = "";
  for (int i = 0; i < v.rows(); i++) {
    s += std::to_string(v(i)) + ",";
  }
  return s;
}

}  // namespace candidate
}  // namespace gpd
