#ifndef LAYER_H_
#define LAYER_H_


class Layer
{
  public:

    virtual ~Layer() { };

    virtual void forward();


  private:

    std::vector<Eigen::MatrixXd> weights_;
    std::vector<Eigen::MatrixXd> inputs_;
    std::vector<Eigen::MatrixXd> outputs_;
};


#endif /* LAYER_H_ */
