#include <convol/convolutional.h>
#include <convol/random.h>
namespace convolutional
{
  Convolutional::Convolutional(OutputInfo info, int featureMaps, int kernelRows,int kernelCols, bool bias, ActivationFunction act,double stdDev, Regularization regularization)
  {}
  OutputInfo Convolutional::initialize(std::vector<double*>& parameterPointers,std::vector<double*>& parameterDerivativePointers)
  {
    OutputInfo info;
    info.dimensions.push_back(fmout);
    outRows = inRows - kernelRows / 2 * 2;
    outCols = inCols - kernelCols / 2 * 2;
    fmOutSize = outRows * outCols;
    info.dimensions.push_back(outRows);
    info.dimensions.push_back(outCols);
    fmInSize = inRows * inCols;
    maxRow = inRows - kernelRows + kernelRows%2;
    maxCol = inCols - kernelCols + kernelCols%2;
    W.resize(fmout, std::vector<Eigen::MatrixXd>(fmin, Eigen::MatrixXd(kernelRows, kernelCols)));
    Wd.resize(fmout, std::vector<Eigen::MatrixXd>(fmin, Eigen::MatrixXd(kernelRows, kernelCols)));
    int numParams = fmout * kernelRows * kernelCols;
    if(bias)
    {
      Wb.resize(fmout, fmin);
      Wbd.resize(fmout, fmin);
      numParams += fmout * fmin;
    }
    parameterPointers.reserve(parameterPointers.size() + numParams);
    parameterDerivativePointers.reserve(parameterDerivativePointers.size() + numParams);
    for(int fmo = 0; fmo < fmout; fmo++)
    {
      for(int fmi = 0; fmi < fmin; fmi++)
      {
        for(int kr = 0; kr < kernelRows; kr++)
        {
          for(int kc = 0; kc < kernelCols; kc++)
          {
            parameterPointers.push_back(&W[fmo][fmi](kr, kc));
            parameterDerivativePointers.push_back(&Wd[fmo][fmi](kr, kc));
          }
        }
        if(bias)
        {
          parameterPointers.push_back(&Wb(fmo, fmi));
          parameterDerivativePointers.push_back(&Wbd(fmo, fmi));
        }
      }
    }

    initializeParameters();

    a.resize(1, info.outputs());
    y.resize(1, info.outputs());
    yd.resize(1, info.outputs());
    deltas.resize(1, info.outputs());

    return info;
  }

  void Convolutional::initializeParameters()
  {
    RandomNumberGenerator rng;
    for(int fmo = 0; fmo < fmout; fmo++)
    {
      for(int fmi = 0; fmi < fmin; fmi++)
      {
        rng.fillNormalDistribution(W[fmo][fmi], stdDev);
        if(bias)
          Wb(fmo, fmi) = rng.sampleNormalDistribution<double>() * stdDev;
      }
    }
  }

  void Convolutional::forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                                       bool dropout, double* error)
  {
    this->x = x;


    const int N = x->rows();
    a.conservativeResize(N, Eigen::NoChange);
    a.setZero();

    for(int n = 0; n < N; n++)
    {
      for(int fmo = 0; fmo < fmout; fmo++)
      {
        int fmInBase = 0;
        for(int fmi = 0; fmi < fmin; fmi++, fmInBase += fmInSize)
        {
          Eigen::MatrixXd& Wtmp = W[fmo][fmi];
          for(int row = 0, outputIdx = fmo * fmOutSize; row < maxRow; row++)
          {
            for(int col = 0; col < maxCol; col++, outputIdx++)
            {
              double& out = a(n, outputIdx);
              for(int kr = 0, colBase = fmInBase + row * inCols + col;
                  kr < kernelRows; kr++, colBase += inCols)
              {
                out += (Wtmp.row(kr).array() *
                    (*x).block(n, colBase, 1, kernelCols).array()).sum();
              }
            }
          }
          if(bias)
          {
            int outputIdx = fmo * fmOutSize;
            for(int row = 0; row < maxRow; row++)
              for(int col = 0; col < maxCol; col++, outputIdx++)
                a(n, outputIdx) += Wb(fmo, fmi);
          }
        }
      }
    }

    this->y.conservativeResize(N, Eigen::NoChange);
    activationFunction(act, a, this->y);

    if(error && regularization.l1Penalty > 0.0)
      for(int fmo = 0; fmo < fmout; fmo++)
        for(int fmi = 0; fmi < fmin; fmi++)
          *error += regularization.l1Penalty * W[fmo][fmi].array().abs().sum();
    if(error && regularization.l2Penalty > 0.0)
      for(int fmo = 0; fmo < fmout; fmo++)
        for(int fmi = 0; fmi < fmin; fmi++)
          *error += regularization.l2Penalty * W[fmo][fmi].array().square().sum() / 2.0;

    y = &(this->y);
  }

  void Convolutional::backpropagate(Eigen::MatrixXd* ein,Eigen::MatrixXd*& eout,bool backpropToPrevious)
  {
    const int N = a.rows();
    // Derive activations
    this->yd.conservativeResize(N, Eigen::NoChange);
    activationFunctionDerivative(act, y, yd);
    deltas = yd.cwiseProduct(*ein);

    e.conservativeResize(N, Eigen::NoChange);
    e.setZero();
    Wbd.setZero();

    for(int fmo = 0; fmo < fmout; fmo++)
      for(int fmi = 0; fmi < fmin; fmi++)
        Wd[fmo][fmi].setZero();

    for(int n = 0; n < N; n++)
    {
      for(int fmo = 0; fmo < fmout; fmo++)
      {
        int fmInBase = 0;
        for(int fmi = 0; fmi < fmin; fmi++, fmInBase += fmInSize)
        {
          Eigen::MatrixXd& Wtmp = W[fmo][fmi];
          Eigen::MatrixXd& Wdtmp = Wd[fmo][fmi];
          for(int row = 0, outputIdx = fmo * fmOutSize; row < maxRow; row++)
          {
            for(int col = 0; col < maxCol; col++, outputIdx++)
            {
              const double d = deltas(n, outputIdx);
              for(int kr = 0, colBase = fmInBase + row * inCols + col;
                  kr < kernelRows; kr++, colBase += inCols)
              {
                e.block(n, colBase, 1, kernelCols) += Wtmp.row(kr) * d;
                Wdtmp.row(kr) += d * (*x).block(n, colBase, 1, kernelCols);
              }
            }
          }
          if(bias)
          {
            for(int row = 0, outputIdx = fmo * fmOutSize; row < maxRow; row++)
              for(int col = 0; col < maxCol; col++, outputIdx++)
                Wbd(fmo, fmi) += deltas(n, outputIdx);
          }
        }
      }
    }

    if(regularization.l1Penalty > 0.0)
    {
      for(int fmo = 0; fmo < fmout; fmo++)
        for(int fmi = 0; fmi < fmin; fmi++)
          Wd[fmo][fmi].array() += regularization.l1Penalty * W[fmo][fmi].array() / W[fmo][fmi].array().abs();
    }
    if(regularization.l2Penalty > 0.0)
    {
      for(int fmo = 0; fmo < fmout; fmo++)
        for(int fmi = 0; fmi < fmin; fmi++)
          Wd[fmo][fmi] += regularization.l2Penalty * W[fmo][fmi];
    }

    eout = &e;
  }

  Eigen::MatrixXd& Convolutional::getOutput()
  {
    return y;
  }

  Eigen::VectorXd Convolutional::getParameters()
  {
    Eigen::VectorXd p(fmout*fmin*kernelRows*kernelCols+bias*fmout*fmin);
    int idx = 0;
    for(int fmo = 0; fmo < fmout; fmo++)
    {
      for(int fmi = 0; fmi < fmin; fmi++)
      {
        for(int kr = 0; kr < kernelRows; kr++)
          for(int kc = 0; kc < kernelCols; kc++)
            p(idx++) = W[fmo][fmi](kr, kc);
      }
    }
    if(bias)
      for(int fmo = 0; fmo < fmout; fmo++)
        for(int fmi = 0; fmi < fmin; fmi++)
            p(idx++) = Wb(fmo, fmi);
    return p;
  }

}
