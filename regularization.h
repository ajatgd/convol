namespace CNN
{
  //regularization helps avoid overfitting as well as the potential of learning extremely large model parameters
  class Regularization
  {
  public:
    //L1 can yield sparse models while L2 doesn't
    double Penaltyl1;
    double Penaltyl2;
    //max value for squared norm of weight vector
    double maxSquaredWeightNorm;
    Regularization(double Penaltyl1=0.0, double Penaltyl2=0.0, double maxSquaredWeightNorm=0.0);
  };
}
