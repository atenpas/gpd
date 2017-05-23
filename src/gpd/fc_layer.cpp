#include <gpd/fc_layer.h>


void FullyConnectedLayer::forward()
{
  outputs_ = weights_ * inputs_;
}
