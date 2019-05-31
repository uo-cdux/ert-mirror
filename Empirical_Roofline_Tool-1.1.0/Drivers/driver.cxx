#include "driver.h"

using namespace std;

Driver::Driver()
  :fMPIRank{0},
  fMPISize{1},
  fNThreads{1},
{
}

Driver::Driver(int argc, char * argv[])
{
#if ERT_GPU
  if (argc != 3) {
    string message = to_string(argv[0]);
    message += " num_blocks num_threads";
    Info::Error(message);
  }

  fGPUBlocks  = atoi(argv[1]);
  fGPUThreads = atoi(argv[2]);
#endif

}

void Driver::Run() {
  int id = 0;
#ifdef ERT_OPENMP
  #pragma omp parallel private(fID)
#endif
  {
#ifdef ERT_OPENMP
    id = omp_get_thread_num();
    nthreads = omp_get_num_threads();
#else
    id = 0;
    nthreads = 1;
#endif
  } // parallel region
}
