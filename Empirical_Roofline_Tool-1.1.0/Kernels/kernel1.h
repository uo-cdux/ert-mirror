#ifndef KERNEL1_H
#define KERNEL1_H

#include <type_traits>
#include <typeinfo>

#include "rep.h"

#ifdef ERT_GPU
#include <cuda_fp16.h>

extern int gpu_blocks;
extern int gpu_threads;

#define KERNEL2HALF(a, b, c) ((a) = __hadd2((b), (c)))
#define KERNEL4HALF(a, b, c) ((a) = __hfma2((a), (b), (c)))
#endif

#ifndef __NVCC__
#define half2 char16_t
#endif

#define KERNEL1(a, b, c) ((a) = (b) + (c))
#define KERNEL2(a, b, c) ((a) = (a) * (b) + (c))

#ifdef ERT_GPU
// If data type is "half2"
template <typename T, typename std::enable_if<std::is_same<T, half2>::value, int>::type = 0>
void initialize(uint64_t nsize, T *__restrict__ A, double value)
{
#if __xlC__
  __alignx(ERT_ALIGN, A);
#endif

  uint64_t i;
  for (i = 0; i < nsize; ++i) {
    A[i] = __float2half2_rn(value);
  }
}
#endif

// If data type is not "half2"
template <typename T, typename std::enable_if<!std::is_same<T, half2>::value, int>::type = 0>
void initialize(uint64_t nsize, T *__restrict__ A, double value)
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
// If data type is "half2"
template <typename T, typename std::enable_if<std::is_same<T, half2>::value, int>::type = 0>
__global__ void block_stride(uint32_t ntrials, uint32_t nsize, T *A)
{
  uint32_t total_thr    = gridDim.x * blockDim.x;
  uint32_t elem_per_thr = (nsize + (total_thr - 1)) / total_thr;
  uint32_t blockOffset  = blockIdx.x * blockDim.x;

  uint32_t start_idx  = blockOffset + threadIdx.x;
  uint32_t end_idx    = start_idx + elem_per_thr * total_thr;
  uint32_t stride_idx = total_thr;

  if (start_idx > nsize) {
    start_idx = nsize;
  }

  if (end_idx > nsize) {
    end_idx = nsize;
  }

  // A needs to be initilized to -1 coming in
  // And with alpha=2 and beta=1, A=-1 is preserved upon return
  T alpha, const_beta;
  alpha      = __float2half2_rn(2.0f);
  const_beta = __float2half2_rn(1.0f);

  uint32_t i, j;
  for (j = 0; j < ntrials; ++j) {
    for (i = start_idx; i < end_idx; i += stride_idx) {
      T beta = const_beta;
#if (ERT_FLOP & 2) == 2 /* add 2 flops */
      KERNEL2HALF(beta, A[i], alpha);
#endif
#if (ERT_FLOP & 4) == 4 /* add 4 flops */
      KERNEL4HALF(beta, A[i], alpha);
#endif
#if (ERT_FLOP & 8) == 8 /* add 8 flops */
      REP2(KERNEL4HALF(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 16) == 16 /* add 16 flops */
      REP4(KERNEL4HALF(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 32) == 32 /* add 32 flops */
      REP8(KERNEL4HALF(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 64) == 64 /* add 64 flops */
      REP16(KERNEL4HALF(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 128) == 128 /* add 128 flops */
      REP32(KERNEL4HALF(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 256) == 256 /* add 256 flops */
      REP64(KERNEL4HALF(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 512) == 512 /* add 512 flops */
      REP128(KERNEL4HALF(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 1024) == 1024 /* add 1024 flops */
      REP256(KERNEL4HALF(beta, A[i], alpha));
#endif

      A[i] = -beta;
    }
  }
}

// If data type is not "half2"
template <typename T, typename std::enable_if<!std::is_same<T, half2>::value, int>::type = 0>
__global__ void block_stride(uint32_t ntrials, uint32_t nsize, T *A)
{
  uint32_t total_thr    = gridDim.x * blockDim.x;
  uint32_t elem_per_thr = (nsize + (total_thr - 1)) / total_thr;

  uint32_t start_idx  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t end_idx    = start_idx + elem_per_thr * total_thr;
  uint32_t stride_idx = total_thr;

  if (start_idx > nsize) {
    start_idx = nsize;
  }

  if (end_idx > nsize) {
    end_idx = nsize;
  }

  // A needs to be initilized to -1 coming in
  // And with alpha=2 and beta=1, A=-1 is preserved upon return
  T alpha, const_beta;
  alpha      = 2.0;
  const_beta = 1.0;

  uint32_t i, j;
  for (j = 0; j < ntrials; ++j) {
    for (i = start_idx; i < end_idx; i += stride_idx) {
      T beta = const_beta;
#if (ERT_FLOP & 1) == 1 /* add 1 flop */
      KERNEL1(beta, A[i], alpha);
#endif
#if (ERT_FLOP & 2) == 2 /* add 2 flops */
      KERNEL2(beta, A[i], alpha);
#endif
#if (ERT_FLOP & 4) == 4 /* add 4 flops */
      REP2(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 8) == 8 /* add 8 flops */
      REP4(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 16) == 16 /* add 16 flops */
      REP8(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 32) == 32 /* add 32 flops */
      REP16(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 64) == 64 /* add 64 flops */
      REP32(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 128) == 128 /* add 128 flops */
      REP64(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 256) == 256 /* add 256 flops */
      REP128(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 512) == 512 /* add 512 flops */
      REP256(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 1024) == 1024 /* add 1024 flops */
      REP512(KERNEL2(beta, A[i], alpha));
#endif

      A[i] = -beta;
    }
  }
}

template <typename T>
void gpuKernel(uint32_t nsize, uint32_t ntrials, T *__restrict__ A, int *bytes_per_elem, int *mem_accesses_per_elem)
{
  *bytes_per_elem        = sizeof(*A);
  *mem_accesses_per_elem = 2;

#ifdef ERT_INTEL
  __assume_aligned(A, ERT_ALIGN);
#elif __xlC__
  __alignx(ERT_ALIGN, A);
#endif

  block_stride<T><<<gpu_blocks, gpu_threads>>>(ntrials, nsize, A);
}
#else
template <typename T>
void kernel(uint64_t nsize, uint64_t ntrials, T *__restrict__ A, int *bytes_per_elem, int *mem_accesses_per_elem)
{
  *bytes_per_elem = sizeof(*A);
  *mem_accesses_per_elem = 2;

#ifdef ERT_INTEL
  __assume_aligned(A, ERT_ALIGN);
#elif __xlC__
  __alignx(ERT_ALIGN, A);
#endif

  T epsilon = 1e-6;
  T factor = (1.0 - epsilon);
  T alpha = -epsilon;
  uint32_t i, j;
  for (j = 0; j < ntrials; ++j) {
#pragma unroll(8)
    for (i = 0; i < nsize; ++i) {
      T beta = A[i] * factor;
#if (ERT_FLOP & 1) == 1 /* add 1 flop */
      KERNEL1(beta, A[i], alpha);
#endif
#if (ERT_FLOP & 2) == 2 /* add 2 flops */
      KERNEL2(beta, A[i], alpha);
#endif
#if (ERT_FLOP & 4) == 4 /* add 4 flops */
      REP2(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 8) == 8 /* add 8 flops */
      REP4(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 16) == 16 /* add 16 flops */
      REP8(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 32) == 32 /* add 32 flops */
      REP16(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 64) == 64 /* add 64 flops */
      REP32(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 128) == 128 /* add 128 flops */
      REP64(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 256) == 256 /* add 256 flops */
      REP128(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 512) == 512 /* add 512 flops */
      REP256(KERNEL2(beta, A[i], alpha));
#endif
#if (ERT_FLOP & 1024) == 1024 /* add 1024 flops */
      REP512(KERNEL2(beta, A[i], alpha));
#endif

      A[i] = beta;
    }
    alpha *= factor;
  }
}
#endif

#endif
