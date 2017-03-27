#include <learner.h>
#include <Eigen/Core>
#include <vector>

namespace CNN
{
  //numerical gradient approxoimation technique
  namespace FiniteDifferences
  {
    //Approximate the derivatives of the error function of a Learner with respect to the inputs numerically.
    //@param X is input,@param y is output
    //@param learner implements an error function
    //@parameps determines precision
    Eigen::MatrixXd inputGradient(const Eigen::MatrixXd& X,const Eigen::MatrixXd& Y, Learner& learner,const double eps = 1e-5);
    //@param n is instance of the training set that is used to calculate the gradient
    //@param opt the optimizable
    //@param eps determines the precision
    Eigen::VectorXd parameterGradient(int n, Optimizable& opt,const double eps = 1e-5);
    /*@param start iterator over mini-batch indices
    @param end iterator over mini-batch indices
    @param opt the optimizable
    @param eps determines the precision*/
    Eigen::VectorXd parameterGradient(std::vector<int>::const_iterator start,std::vector<int>::const_iterator end,Optimizable& opt, const double eps = 1e-5);
  }
}
