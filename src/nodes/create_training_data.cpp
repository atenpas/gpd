// Custom
#include "gpd/data_generator.h"
#include "nodes/ros_params.h"


int main(int argc, char** argv)
{
  // Seed the random number generator.
  std::srand(std::time(0));

  // Initialize ROS.
  ros::init(argc, argv, "create_training_data");
  ros::NodeHandle node("~");

  // Create training data.
  DataGenerator::DataGenerationParameters param;
  ROSParameters::getGeneratorParams(node, param);
  DataGenerator generator(param);
  generator.createTrainingData();

  return 0;
}
