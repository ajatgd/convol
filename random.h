#include <cstdlib>
#include <algorithm>
#include <cmath>
namespace Convolutional
{
  //utility class that simplifies the generation of random numbers. 
  class RandomNumberGenerator
  {
  public:
    //initialize seed
    RandomNumberGenerator();
    //@param seed initial parameter for random number generator
    void seed(unsigned int seed);
    int generateInt(int min, int range) const;
    size_t generateIndex(size_t size) const;

    /*
    Draw a number from a uniform distribution.
    @tparam T number type
    @param min minimal value
    @param range range of the interval, must be greater than 0
    @return random number from the interval [min, range)
    */

    template<class T>
    T generate(T min, T range) const
    {
      if(range == T())
        return min;
      else
        return (T) rand() / (T) RAND_MAX * range + min;
    }
    /*
    RandomNumberGenerator rng;
    double mu = ...
    double sigma = ...
    double rn = mu + sigma*rng.sampleNormalDistribution<double>();
    */
    template<class T>
    T sampleNormalDistribution() const
    {
      return std::sqrt(T(-2) * std::log(generate(T(), T(1)))) *std::cos(T(2) * T(M_PI) * generate(T(), T(1)));
    }

    template<class C>
    void generateIndices(int n, C& result, bool initialized = false)
    {
      if(!initialized)
      {

        for(int i = 0; i < n; i++)
          result.push_back(i);
      }

      std::random_shuffle(result.begin(), result.end());
    }

    template<class M>
    void fillNormalDistribution(M& matrix, double stdDev = 1.0)
    {
      const double* end = matrix.data() + matrix.rows() * matrix.cols();
      for(double* p = matrix.data(); p < end; p++)
          *p = sampleNormalDistribution<double>() * stdDev;
    }
  };
}
