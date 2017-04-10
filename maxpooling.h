#include <layer.h>
#include <activationfunction.h>

namespace CNN
{

/*
 @class MaxPooling
 Performs max-pooling on 2D input feature maps.
 In comparison to average pooling this we have no weights or biases and no
 activation functions in a max-pooling layer. Instead of summing the inputs
 up, it only takes the maximum value. Max-pooling layer are usually more
 efficient than subsampling layers and achieve better results.
 */
class MaxPooling : public Layer
{
  int I, fm, inRows, inCols, kernelRows, kernelCols;
  Eigen::MatrixXd* x;
  Eigen::MatrixXd y;
  Eigen::MatrixXd e;
  int fmInSize, outRows, outCols, fmOutSize, maxRow, maxCol;

public:
  MaxPooling(OutputInfo info, int kernelRows, int kernelCols);
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers, std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters() {}
  virtual void forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y, bool dropout, double* error = 0);
  virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout, bool backpropToPrevious);
  virtual Eigen::MatrixXd& getOutput();
  virtual Eigen::VectorXd getParameters();
};
}
