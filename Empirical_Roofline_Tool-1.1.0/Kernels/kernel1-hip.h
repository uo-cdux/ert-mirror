#ifndef KERNEL1_HIP_H
#define KERNEL1_HIP_H

extern int hip_blocks;
extern int hip_threads;

#define KERNEL1(a,b,c)   ((a) = (b) + (c))
#define KERNEL2(a,b,c)   ((a) = (a)*(b) + (c))

void initialize(uint64_t nsize,
                double* __restrict__ array,
                double value);

void rooflineKernel(uint64_t nsize,
               uint64_t ntrials,
               double* __restrict__ array,
               int* bytes_per_elem,
               int* mem_accesses_per_elem);

#endif // KERNEL1_HIP_H
