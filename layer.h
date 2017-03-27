#include <Eigen/Core>
#include <vector>
namespace CNN
{
  class OutputInfo
  {
  public:
    std::vector<int> dimension;//output dimension
    int outputs();//no. of outputs

  };
  class Layer
  {
  public:
    //return info about output layer. Arguments are pointers to parameter and their derivatives.
    //made virtual so that it can be redefined in derived class

    virtual OutputInfo initialize(std::vector<double*>& parameterPointers, std::vector<double*>& parameterDerivativePointers)=0;
    //initialize the parameters
    virtual void initializeParameters()=0;
    virtual void updatedParameters()=0;
    //forward propogation
    //x points to input for Layer which is a Matrix of doubles
    //y returns pointer to output of layer. It is also a matrix of doubles.
    //dropout enables/disables dropout value for regularization
    //error value that will be updated after regularization
    virtual void forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,bool dropout, double* error=0)=0;
    //backward propagation
    virtual void backPropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,bool backPropToPrevious)=0;
    //output after last forward propagation
    virtual Eigen::MatrixXd& getOutput()=0;

  };
}
