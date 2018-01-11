#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H


#include <Eigen/Dense>

#include <iostream>
#include <vector>

#include <gpd/layer.h>


class DenseLayer : public Layer
{
  public:

    DenseLayer(int num_units) : num_units_(num_units) {}

    Eigen::MatrixXf forward(const std::vector<float>& x) const;


  private:

    int num_units_; ///< the number of units
};


#endif /* DENSE_LAYER_H */
