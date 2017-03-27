#include <convol/convolutional.h>
#include <convol/layeradapter.h>
#include <cstdlib>
#include <convol/random.h>

class ConvTest
{
  virtual void setUp;
  void convolutional();
  void convolutionalGradient();
  void convolutionalInputGradient();
  void regularization();
  void convolutionalGradientWithOddKernel();
};
void ConvTest::setup()
{
  CNN::RandomNumberGenerator rng;
  rng.seed(6);
}
void ConvTest::convolutional()
{
  CNN::OutputInfo info;
  info.dimension.push_back(2);
  info.dimension.push_back(4);
  info.dimension.push_back(4);
  CNN::Convolutional layer(info,2,3,3,false,CNN::TANH,0.05,CNN::Regularization())
  std::vector<double*> pp;
  std::vector<double*> pdp;
  CNN::OutputInfo info2=layer.initialize(pp,pdp);
  ASSERT_EQUALS(info2.dimension.size(),3);//ASSERT_EQUALS referenced from CUTE framework
  ASSERT_EQUALS(info2.dimension[0],2);
  ASSERT_EQUALS(info2.dimension[1],2);
  ASSERT_EQUALS(info2.dimension[2],2);

  for(std::vector<double*>::iterator it == pp.begin(); it!= pp.end(); ++it)
  {
    **it=0.01;
  }
  layer.updatedParameters();
  Eigen::MatrixXd x(1,info.outputs());
  x.fill(1.0);
  Eigen::MatrixXd* y;
  layer.forwardPropagate(&x,y,false);
  ASSERT_EQUALS_DELTA((*y)(0), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(1), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(2), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(3), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(4), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(5), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(6), tanh(0.18), 1e-5);
  ASSERT_EQUALS_DELTA((*y)(7), tanh(0.18), 1e-5);
}

void ConvTest::convolutionalGradient()
{
  CNN::OutputInfo info;
  info.dimension.push_back(3);
  info.dimension.push_back(5);
  info.dimension.push_back(5);
  CNN::Convolutional layer(info,2,3,3,true,CNN::LINEAR,0.05,CNN::Regularization);
  LayerAdapter opt(layer,info);
  Eigen::MatrixXd X=Eigen::MatrixXd::Random(2,3*5*5);
  Eigen::MatrixXd Y=Eigen::MatrixXd::Random(2,2*3*3);
  std::vector<int> indices;
  indices.push_back(0);
  indices.push_back(1);
  opt.trainingSet(X,Y);
  Eigen::VectorXd gradient=opt.gradient(indices.begin(),indices.end());
  Eigen::VectorXd estimatedGradient = CNN::FiniteDifferences::parameterGradient(indices.begin(), indices.end(),opt);
  for(int i=0; i<gradient.rows();i++)
  {
    ASSERT_EQUALS_DELTA(gradient(i),estimatedGradient(i), 1e-10);

  }

}
void ConvTest::convolutionalInputGradient()
{
  CNN::OutputInfo info;
  info.dimension.push_back(3);
  info.dimension.push_back(5);
  info.dimension.push_back(5);
  CNN::Convolutional layer(info,2,3,3,true,CNN::LINEAR,0.05,CNN::Regularization);
  LayerAdapter opt(layer,info);
  Eigen::MatrixXd X=Eigen::MatrixXd::Random(2,3*5*5);
  Eigen::MatrixXd Y=Eigen::MatrixXd::Random(2,2*3*3);
  std::vector<int> indices;
  indices.push_back(0);
  indices.push_back(1);
  opt.trainingSet(X,Y);
  Eigen::MatrixXd gradient=opt.inputGradient();
  ASSERT_EQUALS(gradient.rows(),2);
  Eigen::MatrixXd gradient=opt.inputGradient();
  ASSERT_EQUALS(gradient.rows(),2);
  Eigen::MatrixXd estimatedGradient=CNN::FiniteDifferences::inputGradient(X,Y,opt);
  ASSERT_EQUALS(estimatedGradient.rows(),2);
  for(int j=0;j<gradient.rows();j++)
  {
    for(int i=0; i<gradient.cols();i++)
    {
      ASSERT_EQUALS_DELTA(gradient(j,i),estimatedGradient(j,i), 1e-10);

    }
  }

}
void ConvTest::regularization()
{
  CNN::OutputInfo info;
  info.dimension.push_back(3);
  info.dimension.push_back(5);
  info.dimension.push_back(5);
  CNN::Convolutional layer(info,2,3,3,true,CNN::LINEAR,0.05,CNN::Regularization(0.1,0.1));
  LayerAdapter opt(layer,info);
  Eigen::MatrixXd X=Eigen::MatrixXd::Random(1,3*5*5);
  Eigen::MatrixXd Y=Eigen::MatrixXd::Random(1,2*3*3);
  opt.trainingSet(X,Y);
  Eigen::VectorXd gradient=opt.gradient(0);
  Eigen::VectorXd estimatedGradient=CNN::FiniteDifferences::parameterGradient(0,opt);
  for(int i=0;i<gradient.rows();i++)
  {
    ASSERT_EQUALS_DELTA(gradient(i),estimatedGradient(i), 1e-10);
  }

}
void ConvTest::convolutionalGradientWithOddKernel()
{
  CNN::OutputInfo info;
  info.dimension.push_back(3);
  info.dimension.push_back(4);
  info.dimension.push_back(4);
  CNN::Convolutional layer(info,2,2,2,true,CNN::LINEAR,0.05,CNN::Regularization);
  LayerAdapter opt(layer,info);
  Eigen::MatrixXd X=Eigen::MatrixXd::Random(2,3*4*4);
  Eigen::MatrixXd Y=Eigen::MatrixXd::Random(2,2*2*2);
  std::vector<int> indices;
  indices.push_back(0);
  indices.push_back(1);
  opt.trainingSet(X,Y);
  Eigen::VectorXd gradient=opt.gradient(indices.begin(),indices.end());
  Eigen::VectorXd estimatedGradient=CNN::FiniteDifferences::parameterGradient(indices.begin(),indices.end,opt);
  for(int i=0;i<gradient.rows();i++)
  {
    ASSERT_EQUALS_DELTA(gradient(i),estimatedGradient(i), 1e-10);

  }
}
