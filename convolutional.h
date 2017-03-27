#include "layer.h"
#include "activationfunction.h"
#include "regularization.h"
namespace CNN
{
  class Convolutional : public Layer
  {
    int I, fmin, inRows, inCols, fmout, kernelRows, kernelCols;
    bool bias;
    ActivationFunction act;
    double stdDev;
    Eigen::MatrixXd* x;

    std::vector<std::vector<Eigen::MatrixXd> > W;
    std::vector<std::vector<Eigen::MatrixXd> > Wd;

    Eigen::MatrixXd Wb;
    Eigen::MatrixXd Wbd;
    Eigen::MatrixXd a;
    Eigen::MatrixXd y;
    Eigen::MatrixXd yd;
    Eigen::MatrixXd deltas;
    Eigen::MatrixXd e;
    int fmInSize, outRows, outCols, fmOutSize, maxRow, maxCol;
    Regularization regularization;

  public:
    Convolutional(OutputInfo info, int featureMaps, int kernelRows, int kernelCols, bool bias, ActivationFunction act, double stdDev, Regularization regularization);
    virtual OutputInfo initialize(std::vector<double*>& parameterPointers,std::vector<double*>& parameterDerivativePointers);
    virtual void initializeParameters();
    virtual void updatedParameters() {}
    virtual void forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,bool dropout, double* error = 0);
    virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,bool backpropToPrevious);
    virtual Eigen::MatrixXd& getOutput();
    virtual Eigen::VectorXd getParameters();


  };
}
