// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef TESTS_COMMON_CUH_
#define TESTS_COMMON_CUH_

#include "cuda.h"

// Typed helper functions
template <typename T>
__device__ T floatToType(float val) {
  return (T)val;
}

template <typename T>
__device__ float toFloat(T val) {
  return (T)val;
}

template <>
__device__ half floatToType<half>(float val) {
  return __float2half(val);
}

template <>
__device__ float toFloat<half>(half val) {
  return __half2float(val);
}

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
__device__ __nv_bfloat16 floatToType<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <>
__device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}
#endif

#define DECL_TYPED_KERNS(T)                        \
  template __device__ T floatToType<T>(float val); \
  template __device__ float toFloat<T>(T val);

DECL_TYPED_KERNS(int8_t);
DECL_TYPED_KERNS(uint8_t);
DECL_TYPED_KERNS(int32_t);
DECL_TYPED_KERNS(uint32_t);
DECL_TYPED_KERNS(int64_t);
DECL_TYPED_KERNS(uint64_t);
DECL_TYPED_KERNS(float);
DECL_TYPED_KERNS(double);
// Skip half and __nv_bfloat16 since already declared with specific type

#endif
