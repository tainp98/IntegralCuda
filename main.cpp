#include <iostream>
#include "integralimage.h"

using namespace std;

int main()
{
    IntegralCuda a;
    a.cudaMemoryInit(20,20);
    a.prefixSum2D();
    a.cudaCutsFreeMem();
}

