#include <Eigen/Core>
namespace CNN
{
  //enum of act fn and there derivatives
  //can be added more as required
  enum ActivationFunction
  {
    LOGISTIC=0,
    TANH=1,
    LINEAR=2,
    SOFTMAX=3

  };

  void activationFunction(ActivationFunction act, const Eigen::MatrixXd& a,Eigen::MatrixXd& z);
  void activationFunctionDerivative(ActivationFunction act, const Eigen::MatrixXd& z,Eigen::MatrixXd& gd);
  void softmax(Eigen::MatrixXd& y);
  void logistic(const Eigen::MatrixXd& a,Eigen::MatrixXd& z);
  void logisticDerivative(const Eigen::MatrixXd& z,Eigen::MatrixXd& gd);
  void normaltanh(const Eigen::MatrixXd& a,Eigen::MatrixXd& z);
  void normaltanhDerivative(const Eigen::MatrixXd& z,Eigen::MatrixXd& gd);
  void linear(const Eigen::MatrixXd& a,Eigen::MatrixXd& z);
  void linearDerivative(Eigen::MatrixXd& gd);
}
