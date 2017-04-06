#include <layer.h>
#include <activationfunction.h>
#include <regularization.h>

namespace CNN
{
  /**
  @class Subsampling
  performs average pooling on 2D input feature maps.
  In a subsampling layer non-overlapping regions are
  combined to achieve minor translational invariance
  and to reduce the no of nodes.
  The components of each region will be summed up,
  multiplied by a weight and added to a bias to compute
  the activation of a neuron. Then we apply an
  activation function.
  Supports the following regularization types:
 - L1 penalty
 - L2 penalty
 */

  class Subsampling : public layer
  {
    int I, fm, inRows, inCols, kernelRows, kernelCols;
    bool bias;
    ActivationFunction act;
    double stdDev;
    Eigen::MatrixXd* x;
    // feature maps X output rows X output cols
    std::vector<Eigen::MatrixXd> W;
    std::vector<Eigen::MatrixXd> Wd;
    // feature maps X output rows X output cols
    std::vector<Eigen::MatrixXd> Wb;
    std::vector<Eigen::MatrixXd> Wbd;
    Eigen::MatrixXd a;
    Eigen::MatrixXd y;
    Eigen::MatrixXd yd;
    Eigen::MatrixXd deltas;
    Eigen::MatrixXd e;
    int fmInSize, outRows, outCols, fmOutSize, maxRow, maxCol;
    Regularization regularization;
  public:
    Subsampling(OutputInfo info, int kernelRows, int kernelCols, bool bias, ActivationFunction act, double stdDev, Regularization regularization);
    virtual OutputInfo initialize(std::vector<double*>& parameterPointers,std::vector<double*>& parameterDerivativePointers);
    virtual void initializeParameters();
    virtual void updatedParameters(){};
    virtual void forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd* &y, bool dropout, double* error=0);
    virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout, bool backpropToPrevious);
    virtual Eigen::MatrixXd& getOutput();
    virtual Eigen::VectorXd getParameters();
  };
}
