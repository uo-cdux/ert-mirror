#include "Array.h"

#include <iostream>

using namespace std;

Array::Array (uint64_t size)
{
  this->Alloc(uint64_t size);
}

Array::Array (int argc, char *argv[])
{
#ifndef ERT_MPI
  string message = "This constructor cannot be called without ERT_MPI";
  Info::Error(message);
#endif
  
  fMPIProvided = -1;

#ifdef ERT_OPENMP
  fMPIRequested = MPI_THREAD_FUNNELED;
  MPI_Init_thread(&argc, &argv, fMPIRequested, &fMPIProvided);
#else
  MPI_Init(&argc, &argv);
#endif

  MPI_Comm_size(MPI_COMM_WORLD, &fMPISize);
  MPI_Comm_rank(MPI_COMM_WORLD, &fMPIRank);

  this->Alloc();
}

void Array::Alloc (uint64_t size) {
  uint64_t psize = ERT_MEMORY_MAX / size;

#ifdef ERT_INTEL
  fBuffer = (T *) _mm_malloc(psize, ERT_ALIGN);
#else
  fBuffer = (T *) malloc(psize);
#endif

  if (fBuffer == nullptr) {
    string message = "Out of memory";
    Error(message);
  }
}
