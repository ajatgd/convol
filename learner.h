#include "optimizable.h"'
namespace CNN
{
 class learner
 {
   //dataset will be loaded here
   //training and testing data will be specified

 public:
   //Set training set
   //instances of input and output
   virtual learner& trainingSet(Eigen::MatrixXd& input, Eigen::MatrixXd& output);

 };
}
