#include <CL/cl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>
#include <inttypes.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>

#ifndef DEVICE
#  define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

inline std::string loadProgram(std::string input)
{   
    std::ifstream stream(input.c_str());
    if (!stream.is_open()) {
        std::cout << "Cannot open file: " << input << std::endl;
        exit(1);
    }
     
     return std::string(
        std::istreambuf_iterator<char>(stream),
        (std::istreambuf_iterator<char>()));
}

void initialize(uint64_t nsize,
                double* __restrict__ A,
                double value)
{
  uint64_t i;
  for (i = 0; i < nsize; ++i) {
    A[i] = value;
  }
}

double getTime()
{
  double time;

  struct timeval tm;
  gettimeofday(&tm, NULL);
  time = tm.tv_sec + (tm.tv_usec / 1000000.0);

  return time;
}

int main(int argc, char *argv[]) {

  int rank = 0;
  int nprocs = 1;
  int id = 0;

  double startTime, endTime;
  uint64_t n;
  uint64_t t;

  uint64_t TSIZE = ERT_MEMORY_MAX;
  uint64_t PSIZE = TSIZE / nprocs;

  uint64_t nsize = PSIZE;
  nsize = nsize & (~(ERT_ALIGN-1));
  nsize = nsize / sizeof(double);
  uint64_t nid =  nsize * id ;

  std::vector<double> buf(PSIZE/sizeof(double));
  std::vector<double> bufv(PSIZE/sizeof(double));
  std::vector<int> params(3);
  uint32_t ert_flops;
  uint wg_size;

  cl::Buffer d_buf, d_params;
  cl::Context context(DEVICE);
  cl::CommandQueue queue(context);
  cl::Program program(context, loadProgram("Kernels/kernel1.cl"), false);
  char build_args[80];
  sprintf(build_args, "-DERT_FLOP=%d", ERT_FLOP);
  if (program.build(build_args) != CL_SUCCESS) {
    fprintf(stderr, "ERROR: kernel.cl failed to build\n");
    exit(-1);
  }
  auto ocl_kernel = cl::make_kernel<ulong, cl::Buffer, cl::Buffer>(program, "ocl_kernel");

  std::vector<cl::Device> devices;
  context.getInfo(CL_CONTEXT_DEVICES, &devices); 
  cl::Device device=devices[0];

  if (argc == 2) {
    uint max_wg_size;
    max_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    wg_size = atoi(argv[1]);
    if (wg_size > max_wg_size)
      wg_size = max_wg_size;
  }
  else {
    wg_size=0;  // let the OpenCL runtime decide
  }

#ifdef HEADER
  {
    std::string s, v;

    device.getInfo(CL_DEVICE_VENDOR, &v);
    device.getInfo(CL_DEVICE_NAME, &s);
    std::cout << "Using device: " << v << ": " << s << std::endl;

    printf("%12s %12s %15s %12s %12s %12s %12s\n", 
      "nsize", "trials", "microsecs", "bytes", "flops", "GB/s", "GF/s");
  }
#endif

  n = ERT_WORKING_SET_MIN;
  while (n <= nsize) { // working set - nsize

    // number of trials derived from global memory size and working set size
    uint64_t ntrials = nsize / n;
    if (ntrials < 1)
      ntrials = 1;

    // initialize the data buffer
    initialize(nsize, &buf[nid], -1.0);

    // define parameters on the device
    d_params = cl::Buffer(context, begin(params), end(params), true);
    // copy working set buffer to device
    d_buf = cl::Buffer(context, begin(buf), begin(buf)+n, true);

    // loop through increasing numbers of trials
    for (t = ERT_TRIALS_MIN; t <= ntrials; t = t * 2) { // working set - ntrials

      // queue up trials
      // note: currently OCL kernel does not account for nid offset
      startTime = getTime();
      if (wg_size > 0)    // local working group size is set
          ocl_kernel(cl::EnqueueArgs(queue, cl::NDRange(n), cl::NDRange(wg_size)), t, d_buf, d_params);
      else                // let the runtime set local working group size
          ocl_kernel(cl::EnqueueArgs(queue, cl::NDRange(n)), t, d_buf, d_params);
      queue.finish();  // wait for all trials to finish
      endTime = getTime();

      // print out the resuls
      if ((id == 0) && (rank == 0)) {
        int bytes_per_elem;
        int mem_accesses_per_elem;

        cl::copy(queue, d_params, begin(params), end(params));
        bytes_per_elem = params[0];
        mem_accesses_per_elem = params[1];
        ert_flops = params[2];
        double seconds = (double)(endTime - startTime);
        uint64_t working_set_size = n * nprocs;
        uint64_t total_bytes = t * working_set_size * bytes_per_elem * mem_accesses_per_elem;
        uint64_t total_flops = t * working_set_size * ert_flops;
        printf("%12lu %12lu %15.3lf %12lu %12lu\n",
                working_set_size * bytes_per_elem,
                t,
                seconds * 1000000,
                total_bytes,
                total_flops);
      } // print

#ifdef VERIFY
      {
        // as a runtime optimization, there's no need to copy data back unless debuging
        // as this is outside the timer loop
        cl::copy(queue, d_buf, begin(bufv), begin(bufv)+n);
        for (int i=0; i<n; ++i) {
          if (bufv[i] != (double)-1.0) {
            printf("ERROR: bufv[%d] == %e != -1.0\n", i, bufv[i]);
            break;
          }
        }
      }
#endif
    } // working set - ntrials

//    n *= 1.1;
    n *= 2.0;

  } // working set - nsize

  printf("\n");
  printf("META_DATA\n");
  printf("FLOPS          %d\n", ert_flops);
  if (wg_size > 0)
    printf("WGSIZE         %d\n", wg_size);
  else
    printf("WGSIZE         NULL\n");

  return 0;
}
