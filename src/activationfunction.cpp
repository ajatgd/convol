#include <convol/activationfunction.h>
#include <limits>
#include <cmath>
namespace CNN
{
  void activationfunction(ActivationFunction act, const Eigen::MatrixXd& a,Eigen::MatrixXd& z)
  {
    switch(act)
    {
      case LOGISTIC:
        logistic(a,z);
        break;
      case TANH:
        normaltanh(a,z);
        break;
      case LINEAR:
      default:
        linear(a,z);
        break;
    }
  }
  void activationFunctionDerivative(ActivationFunction act, const Eigen::MatrixXd& z, Eigen::MatrixXd& gd)
  {
    switch(act)
    {
      case LOGISTIC:
        logisticDerivative(z,gd);
        break;
      case TANH:
        normaltanhDerivative(z,gd);
        break;
      case LINEAR:
      default:
        linearDerivative(gd);
        break;

    }
  }
  void softmax(Eigen::MatrixXd& y)
  {
    const int N=y.rows();
    const double max=y.maxCoeff();
    for(int n=0;n<N;n++)
    {
      y.row(n)=(y.row(n).array()-max).exp();
      y.row(n)/=y.row(n).sum();

    }
  }
  void logistic(const Eigen::MatrixXd& a, Eigen::MatrixXd& z)
  {
    double const* aPtr = a.data();
    double const* aEnd = aPtr + a.rows() * a.cols();
    for(double* zPtr = z.data(); aPtr < aEnd; aPtr++, zPtr++)
    {
      if(*aPtr < -45.0)
        *zPtr = 0.0;
      else if(*aPtr > 45.0)
        *zPtr = 1.0;
      else
        *zPtr = 1.0 / (1.0 + std::exp(-*aPtr));
    }
  }

  void logisticDerivative(const Eigen::MatrixXd& z, Eigen::MatrixXd& gd)
  {
    double const* zPtr = z.data();
    double const* zEnd = zPtr + z.rows() * z.cols();
    for(double* gdPtr = gd.data(); zPtr < zEnd; zPtr++, gdPtr++)
      *gdPtr = *zPtr * (1.0 - *zPtr);
  }

  void normaltanh(const Eigen::MatrixXd& a, Eigen::MatrixXd& z)
  {
    double const* aPtr = a.data();
    double const* aEnd = aPtr + a.rows() * a.cols();
    for(double* zPtr = z.data(); aPtr < aEnd; aPtr++, zPtr++)
      *zPtr = std::tanh(*aPtr);
  }

  void normaltanhDerivative(const Eigen::MatrixXd& z, Eigen::MatrixXd& gd)
  {
    double const* zPtr = z.data();
    double const* zEnd = zPtr + z.rows() * z.cols();
    for(double* gdPtr = gd.data(); zPtr < zEnd; zPtr++, gdPtr++)
      *gdPtr = 1.0 - *zPtr * *zPtr;
  }
  void linear(const Eigen::MatrixXd& a, Eigen::MatrixXd& z)
  {
    z = a;
  }

  void linearDerivative(Eigen::MatrixXd& gd)
  {
    gd.fill(1.0);
  }

}
