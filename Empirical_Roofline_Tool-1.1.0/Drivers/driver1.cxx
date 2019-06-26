#include "driver.h"
#include "kernel1.h"

double getTime()
{
  double time;

#ifdef ERT_OPENMP
  time = omp_get_wtime();
#elif ERT_MPI
  time = MPI_Wtime();
#else
  struct timeval tm;
  gettimeofday(&tm, NULL);
  time = tm.tv_sec + (tm.tv_usec / 1000000.0);
#endif
  return time;
}

template <typename T>
inline void checkBuffer(T *buffer) {
  if (buffer == nullptr) {
    fprintf(stderr, "Out of memory!\n");
    exit(1);
  }
} 

template <typename T>
T* alloc(uint64_t psize) {
#ifdef ERT_INTEL
  return  (T *)_mm_malloc(psize, ERT_ALIGN);
#else
  return (T *)malloc(psize);
#endif
}

template <typename T>
T* setDeviceData(uint64_t nsize) {
  T* buf;
  cudaMalloc((void **)&buf, nsize*sizeof(T));
  cudaMemset(buf, 0, nsize*sizeof(T));
  return buf;
}

inline void setGPU(const int id) {
  int num_gpus = 0;
  int gpu;
  int gpu_id;
  int numSMs;

  cudaGetDeviceCount(&num_gpus);
  if (num_gpus < 1) {
    fprintf(stderr, "No CUDA device detected.\n");
    exit(1);
  }

  for (gpu = 0; gpu < num_gpus; gpu++) {
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop,gpu);
  }

  cudaSetDevice(id % num_gpus);
  cudaGetDevice(&gpu_id);
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, gpu_id);
}

template <typename T>
inline void launchKernel(
  uint64_t n,
  uint64_t t,
  uint64_t nid,
  T *buf,
  T *d_buf,
  int *bytes_per_elem_ptr,
  int *mem_accesses_per_elem_ptr)
{
#if    ERT_AVX // AVX intrinsics for Edison(intel xeon)
  avxKernel(n, t, &buf[nid]);
#elif  ERT_KNC // KNC intrinsics for Babbage(intel mic)
  kncKernel(n, t, &buf[nid]);
#elif  ERT_GPU // CUDA code
  gpuKernel<T>(n, t, d_buf, bytes_per_elem_ptr, mem_accesses_per_elem_ptr);
  cudaDeviceSynchronize();
#else          // C-code
  kernel<T>(n, t, &buf[nid], bytes_per_elem_ptr, mem_accesses_per_elem_ptr);
#endif
}

template <typename T>
void run(uint64_t PSIZE, T* buf, int rank, int nprocs)
{
  if (std::is_floating_point<T>::value) {
    if (sizeof(T) == 4) {
      printf("single\n");
    }
    else if (sizeof(T) == 8) {
      printf("double\n");
    }
  }
  else if (std::is_same<T, half2>::value) {
    printf("half\n");
  }
  else {
    fprintf(stderr, "Data type not supported.\n");
    exit(1);
  }
  int nthreads = 1;
  int id = 0;
#ifdef ERT_OPENMP
  #pragma omp parallel private(id)
#endif

  {
#ifdef ERT_OPENMP
    id = omp_get_thread_num();
    nthreads = omp_get_num_threads();
#endif

#if ERT_GPU
    setGPU(id);
#endif
        
    uint64_t nsize = PSIZE / nthreads;
    nsize = nsize & (~(ERT_ALIGN-1));
    nsize = nsize / sizeof(T);
    uint64_t nid = nsize * id ;

    // initialize small chunck of buffer within each thread
    double value = 1.0;
    initialize<T>(nsize, &buf[nid], value);

#if ERT_GPU
    T *d_buf = setDeviceData<T>(nsize);
    cudaDeviceSynchronize();
#endif

#ifdef ERT_GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#else
    double startTime, endTime;
#endif
    uint64_t n,nNew;
    uint64_t t;
    int bytes_per_elem;
    int mem_accesses_per_elem;

    n = ERT_WORKING_SET_MIN;
    while (n <= nsize) { // working set - nsize

      uint64_t ntrials = nsize / n;
      if (ntrials < 1)
        ntrials = 1;

      for (t = ERT_TRIALS_MIN; t <= ntrials; t = t * 2) { // working set - ntrials
#ifdef ERT_GPU
        cudaMemcpy(d_buf, &buf[nid], n*sizeof(T), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
#endif

#ifdef ERT_MPI
  #ifdef ERT_OPENMP        
        #pragma omp master
  #endif
        {
          MPI_Barrier(MPI_COMM_WORLD);
        }
#endif // ERT_MPI

#ifdef ERT_OPENMP
        #pragma omp barrier
#endif

        if ((id == 0) && (rank==0)) {
#ifdef ERT_GPU
          cudaEventRecord(start);
#else
          startTime = getTime();
#endif
        }
        
        launchKernel<T>(n, t, nid, buf, d_buf, &bytes_per_elem, &mem_accesses_per_elem);

#ifdef ERT_OPENMP
        #pragma omp barrier
#endif

#ifdef ERT_MPI
  #ifdef ERT_OPENMP
        #pragma omp master
  #endif
        {
          MPI_Barrier(MPI_COMM_WORLD);
        }
#endif // ERT_MPI
        
        if ((id == 0) && (rank==0)) {
#ifdef ERT_GPU
          cudaEventRecord(stop);
#else
          endTime = getTime();
#endif
        }

#if ERT_GPU
        cudaMemcpy(&buf[nid], d_buf, n*sizeof(T), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
#endif

        if ((id == 0) && (rank == 0)) {
          uint64_t working_set_size = n * nthreads * nprocs;
          uint64_t total_bytes = t * working_set_size * bytes_per_elem * mem_accesses_per_elem;
          uint64_t total_flops = t * working_set_size * ERT_FLOP;
          double seconds;

          // nsize; trials; microseconds; bytes; single thread bandwidth; total bandwidth
#if ERT_GPU
          cudaEventSynchronize(stop);
          float milliseconds = 0.f;
          cudaEventElapsedTime(&milliseconds, start, stop);
          seconds = static_cast<double>(milliseconds) / 1000.;
#else
          endTime = getTime();
          seconds = endTime - startTime;
#endif
          printf("%12" PRIu64 " %12" PRIu64 " %15.3lf %12" PRIu64 " %12" PRIu64 "\n",
                  working_set_size * bytes_per_elem,
                  t,
                  seconds * 1000000,
                  total_bytes,
                  total_flops);
        } // print

      } // working set - ntrials

      nNew = 1.1 * n;
      if (nNew == n) {
        nNew = n+1;
      }

      n = nNew;
    } // working set - nsize

#if ERT_GPU
    cudaFree(d_buf);

    if (cudaGetLastError() != cudaSuccess) {
      printf("Last cuda error: %s\n",cudaGetErrorString(cudaGetLastError()));
    }

    cudaDeviceReset();
#endif
  } // parallel region
}

int main(int argc, char *argv[]) {
#if ERT_GPU
  if (argc != 3) {
    fprintf(stderr, "Usage: %s gpu_blocks gpu_threads\n", argv[0]);
    return -1;
  }

  gpu_blocks  = atoi(argv[1]);
  gpu_threads = atoi(argv[2]);
#endif

  int rank = 0;
  int nprocs = 1;
#ifdef ERT_MPI
  int provided = -1;
  int requested;

  #ifdef ERT_OPENMP
  requested = MPI_THREAD_FUNNELED;
  MPI_Init_thread(&argc, &argv, requested, &provided);
  #else
  MPI_Init(&argc, &argv);
  #endif // ERT_OPENMP

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* printf("The MPI binding provided thread support of: %d\n", provided); */
#endif // ERT_MPI

  uint64_t TSIZE = ERT_MEMORY_MAX;
  uint64_t PSIZE = TSIZE / nprocs;

#if ERT_GPU
  half2 *              hlfbuf = alloc<half2>(PSIZE);
  double *              dblbuf = alloc<double>(PSIZE);
  float *              sglbuf = alloc<float>(PSIZE);
#else
  double * __restrict__ dblbuf = alloc<double>(PSIZE);
  float * __restrict__ sglbuf = alloc<float>(PSIZE);
#endif

#if ERT_GPU
  checkBuffer(hlfbuf);
#endif
  checkBuffer(dblbuf);
  checkBuffer(sglbuf);

#if ERT_GPU
  run<half2>(PSIZE, hlfbuf, rank, nprocs);
#endif
  run<double>(PSIZE, dblbuf, rank, nprocs);
  run<float>(PSIZE, sglbuf, rank, nprocs);

#ifdef ERT_INTEL
  _mm_free(dblbuf);
  _mm_free(sglbuf);
#elif ERT_GPU
  free(hlfbuf);
#else
  free(dblbuf);
  free(sglbuf);
#endif

#ifdef ERT_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef ERT_MPI
  MPI_Finalize();
#endif

  printf("\n");
  printf("META_DATA\n");
  printf("FLOPS          %d\n", ERT_FLOP);

#ifdef ERT_MPI
  printf("MPI_PROCS      %d\n", nprocs);
#endif

#ifdef ERT_OPENMP
  printf("OPENMP_THREADS %d\n", nthreads);
#endif

#ifdef ERT_GPU
  printf("GPU_BLOCKS     %d\n", gpu_blocks);
  printf("GPU_THREADS    %d\n", gpu_threads);
#endif

  return 0;
}
