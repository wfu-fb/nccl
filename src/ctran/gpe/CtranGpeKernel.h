// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_KERNEL_H_
#define CTRAN_GPE_KERNEL_H_

#include <stdint.h>

#define UNSET (0)
#define KERNEL_STARTED (1)
#define KERNEL_TERMINATE (2)

inline __device__ int loadInt(int* ptr) {
  int v;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
}

inline __device__ void storeInt(int* ptr, int val) {
  asm volatile("st.volatile.global.s32 [%0], %1;" ::"l"(ptr), "r"(val));
}

static inline __device__ void ncclKernelStartGpe(int* flag) {
  storeInt(flag, KERNEL_STARTED);
}

static inline __device__ void ncclKernelWaitGpeTerminate(int* flag) {
  int flagVal = KERNEL_STARTED;
  do {
    flagVal = loadInt(flag);
  } while (flagVal != KERNEL_TERMINATE);
}
#endif
