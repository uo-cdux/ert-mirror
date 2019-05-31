#ifndef KERNEL1_H
#define KERNEL1_H

#ifdef ERT_GPU
extern int gpu_blocks;
extern int gpu_threads;
#endif

#define KERNEL1(a,b,c)   ((a) = (b) + (c))
#define KERNEL2(a,b,c)   ((a) = (a)*(b) + (c))

template <typename T>
void initialize(uint64_t nsize,
                T* __restrict__ A,
                T value)
{
#ifdef ERT_INTEL
  __assume_aligned(A, ERT_ALIGN);
#elif __xlC__
  __alignx(ERT_ALIGN, A);
#endif

  uint64_t i;
  for (i = 0; i < nsize; ++i) {
    A[i] = value;
  }
}

#ifdef ERT_GPU
void gpuKernel(uint64_t nsize,
               uint64_t ntrials,
               double* __restrict__ array,
               int* bytes_per_elem,
               int* mem_accesses_per_elem);
#else
void kernel(uint64_t nsize,
            uint64_t ntrials,
            double* __restrict__ array,
            int* bytes_per_elem,
            int* mem_accesses_per_elem);
#endif

#endif
