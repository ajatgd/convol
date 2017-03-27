#include <convol/random.h>
#include <ctime>

namespace CNN
{

RandomNumberGenerator::RandomNumberGenerator()
{
  static bool seedInitialized = false;
  if(!seedInitialized)
  {
    srand(std::time(0));
    seedInitialized = true;
  }
}

void RandomNumberGenerator::seed(unsigned int seed)
{
  srand(seed);
}

int RandomNumberGenerator::generateInt(int min, int range) const
{
  if(range == 0)
    return min;
  else
    return rand() % range + min;
}

size_t RandomNumberGenerator::generateIndex(size_t size) const
{
  return (size_t) generateInt(0, size);
}

}
