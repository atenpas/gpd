// Custom
#include "../../include/gpd/data_generator.h"


int main(int argc, char** argv)
{
  // Seed the random number generator.
  std::srand(std::time(0));

  // Initialize ROS.
  ros::init(argc, argv, "create_training_data");
  ros::NodeHandle node("~");

  // Create training data.
  DataGenerator generator(node);
  generator.createTrainingData(node);

  return 0;
}
