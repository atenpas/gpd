#ifndef FC_LAYER_H_
#define FC_LAYER_H_


#include <gpd/layer.h>


class FullyConnectedLayer : public Layer
{
  public:

    void forward();
};


#endif /* FC_LAYER_H_ */
