#include "layer.h"
#include "Eigen/Core"
#include "learner"

class LayerAdapter
{
  CNN::Layer& layer;
  std::vector<double*> parameters;
  std::vector<double*> derivatives;
  Eigen::MatrixXd input;
  Eigen::MatrixXd desired;
  Eigen::VectorXd params;
  CNN::OutputInfo info;
public:
  LayerAdapter(CNN::Layer& layer, CNN::OutputInfo inputs);
  virtual Eigen::VectorXd gradient();
  virtual Eigen::VectorXd gradient(unsigned int n);
  virtual Eigen::VectorXd gradient(std::vector<int>::const_iterator startN,std::vector<int>::const_iterator endN);
  Eigen::MatrixXd inputGradient();
  virtual Learner& trainingSet(Eigen::MatrixXd& trainingInput,Eigen::MatrixXd& trainingOutput);
  virtual Learner& trainingSet(CNN::DataSet& trainingSet);
}
