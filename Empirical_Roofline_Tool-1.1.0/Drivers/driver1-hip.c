// HIP verson of driver1.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <inttypes.h>
#include <hip/hip_runtime.h>
#ifdef ERT_OPENMP
  #include <omp.h>
#endif
#include "kernel1-hip.h"


double getTime()
{
  double time;

#ifdef ERT_OPENMP
  time = omp_get_wtime();
#else
  struct timeval tm;
  gettimeofday(&tm, NULL);
  time = tm.tv_sec + (tm.tv_usec / 1000000.0);
#endif
  return time;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s hip_blocks hip_threads\n", argv[0]);
    return -1;
  }

  hip_blocks  = atoi(argv[1]);
  hip_threads = atoi(argv[2]);

  int rank = 0;
  int nprocs = 1;
  int nthreads = 1;
  int id = 0;

  uint64_t TSIZE = ERT_MEMORY_MAX;
  uint64_t PSIZE = TSIZE / nprocs;

  double *buf;
  if (posix_memalign((void **)&buf, ERT_ALIGN, PSIZE) != 0) {
    fprintf(stderr, "Out of memory!\n");
    return -1;
  }

#ifdef ERT_OPENMP
  #pragma omp parallel private(id)
#endif
  {
#ifdef ERT_OPENMP
    id = omp_get_thread_num();
    nthreads = omp_get_num_threads();
#else
    id = 0;
    nthreads = 1;
#endif

    int num_devices = 0;
    int device;
    int device_id;
    int numSMs;

    hipGetDeviceCount(&num_devices);
    if (num_devices < 1) {
      fprintf(stderr, "No devices detected.\n");
      return -1;
    }

    for (device = 0; device < num_devices; device++) {
      hipDeviceProp_t dprop;
      hipGetDeviceProperties(&dprop,device);
    }

    hipSetDevice(id % num_devices);
    hipGetDevice(&device_id);
    hipDeviceGetAttribute(&numSMs, hipDeviceAttributeMultiprocessorCount, device_id);
        
    uint64_t nsize = PSIZE / nthreads;
    nsize = nsize / sizeof(double);
    uint64_t nid =  nsize * id ;

    double *d_buf;
    hipMalloc((void **)&d_buf, nsize*sizeof(double));
    hipMemset(d_buf, 0, nsize*sizeof(double));
    hipDeviceSynchronize();

    double startTime, endTime;
    uint64_t n,nNew;
    uint64_t t;
    int bytes_per_elem;
    int mem_accesses_per_elem;

    n = ERT_WORKING_SET_MIN;
    while (n <= nsize) { // working set - nsize
      uint64_t ntrials = nsize / n;
      if (ntrials < 1)
        ntrials = 1;

      // initialize small chunck of buffer within each thread
      initialize(n, &buf[nid], -1.0);

      for (t = ERT_TRIALS_MIN; t <= ntrials; t = t * 2) { // working set - ntrials
        hipMemcpy(d_buf, &buf[nid], n*sizeof(double), hipMemcpyHostToDevice);
        hipDeviceSynchronize();

#ifdef ERT_OPENMP
        #pragma omp barrier
#endif
        if ((id == 0) && (rank==0)) {
          startTime = getTime();
        }

        rooflineKernel(n, t, d_buf, &bytes_per_elem, &mem_accesses_per_elem);
        hipDeviceSynchronize();
#ifdef ERT_OPENMP
        #pragma omp barrier
#endif

        if ((id == 0) && (rank == 0)) {
          endTime = getTime();
          double seconds = (double)(endTime - startTime);
          uint64_t working_set_size = n * nthreads * nprocs;
          uint64_t total_bytes = t * working_set_size * bytes_per_elem * mem_accesses_per_elem;
          uint64_t total_flops = t * working_set_size * ERT_FLOP;

          // nsize; trials; microseconds; bytes; single thread bandwidth; total bandwidth
          printf("%12lu %12ld %15.3lf %12lu %12lu\n",
                  working_set_size * bytes_per_elem,
                  t,
                  seconds * 1000000,
                  total_bytes,
                  total_flops);
        } // print

        hipMemcpy(&buf[nid], d_buf, n*sizeof(double), hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
      } // working set - ntrials

      nNew = 1.1 * n;
      if (nNew == n) {
        nNew = n+1;
      }

      n = nNew;
    } // working set - nsize

    hipFree(d_buf);

    if (hipGetLastError() != hipSuccess) {
      printf("Last cuda error: %s\n",hipGetErrorString(hipGetLastError()));
    }

    hipDeviceReset();
  } // parallel region

  free(buf);

  printf("\n");
  printf("META_DATA\n");
  printf("FLOPS          %d\n", ERT_FLOP);
#ifdef ERT_OPENMP
  printf("OPENMP_THREADS %d\n", nthreads);
#endif
  printf("GPU_BLOCKS     %d\n", hip_blocks);
  printf("GPU_THREADS    %d\n", hip_threads);

  return 0;
}
