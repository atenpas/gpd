#ifndef LAYER_H
#define LAYER_H


#include <Eigen/Dense>

#include <vector>


class Layer
{
  public:

    virtual ~Layer() {}

    virtual Eigen::MatrixXf forward(const std::vector<float>& x) const = 0;

    void setWeightsAndBiases(const std::vector<float>& weights, const std::vector<float>& biases)
    {
      weights_ = weights;
      biases_ = biases;
    }


  protected:

    std::vector<float> weights_;
    std::vector<float> biases_;
};


#endif /* LAYER_H */
