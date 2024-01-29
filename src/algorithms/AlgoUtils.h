#pragma once

#include "nccl.h"
#include <stdint.h>

struct DdaDeviceState {
  uintptr_t* threadedBarrierMbox;
  uintptr_t* ipcBarrierMbox;
  void* tmpbuff;
};

namespace nccl {
namespace algorithms {

#define ASSIGN_FUNC_NRANKS(func, templ, nranks) \
  do {                                          \
    switch ((nranks)) {                         \
      case 2:                                   \
        func = (const void*)&templ<T, 2>;       \
        break;                                  \
                                                \
      case 4:                                   \
        func = (const void*)&templ<T, 4>;       \
        break;                                  \
                                                \
      case 8:                                   \
        func = (const void*)&templ<T, 8>;       \
        break;                                  \
                                                \
      case 16:                                  \
        func = (const void*)&templ<T, 16>;      \
        break;                                  \
                                                \
      default:                                  \
        return ncclInvalidUsage;                \
    }                                           \
  } while (0)

// Need a better a way to do this?
#if defined(__CUDA_BF16_TYPES_EXIST__) && defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
#define NCCL_TYPED_CALL(ncclDataType, func, ...)    \
  ({                                                \
    ncclResult_t __res;                             \
    do {                                            \
      switch (ncclDataType) {                       \
        case ncclInt8: {                            \
          __res = func<char>(__VA_ARGS__);          \
          break;                                    \
        }                                           \
        case ncclUint8: {                           \
          __res = func<uint8_t>(__VA_ARGS__);       \
          break;                                    \
        }                                           \
        case ncclInt32: {                           \
          __res = func<int32_t>(__VA_ARGS__);       \
          break;                                    \
        }                                           \
        case ncclUint32: {                          \
          __res = func<uint32_t>(__VA_ARGS__);      \
          break;                                    \
        }                                           \
        case ncclInt64: {                           \
          __res = func<int64_t>(__VA_ARGS__);       \
          break;                                    \
        }                                           \
        case ncclUint64: {                          \
          __res = func<uint64_t>(__VA_ARGS__);      \
          break;                                    \
        }                                           \
        case ncclFloat16: {                         \
          __res = func<half>(__VA_ARGS__);          \
          break;                                    \
        }                                           \
        case ncclFloat32: {                         \
          __res = func<float>(__VA_ARGS__);         \
          break;                                    \
        }                                           \
        case ncclFloat64: {                         \
          __res = func<double>(__VA_ARGS__);        \
          break;                                    \
        }                                           \
        case ncclBfloat16: {                        \
          __res = func<__nv_bfloat16>(__VA_ARGS__); \
          break;                                    \
        }                                           \
        case ncclFp8E5M2: {                        \
          __res = func<__nv_fp8_e5m2>(__VA_ARGS__); \
          break;                                    \
        }                                           \
        case ncclFp8E4M3: {                        \
          __res = func<__nv_fp8_e4m3>(__VA_ARGS__); \
          break;                                    \
        }                                           \
        default: {                                  \
          __res = ncclInvalidArgument;              \
        }                                           \
      }                                             \
    } while (0);                                    \
    __res;                                          \
  })
#elif defined(__CUDA_BF16_TYPES_EXIST__)
#define NCCL_TYPED_CALL(ncclDataType, func, ...)    \
  ({                                                \
    ncclResult_t __res;                             \
    do {                                            \
      switch (ncclDataType) {                       \
        case ncclInt8: {                            \
          __res = func<char>(__VA_ARGS__);          \
          break;                                    \
        }                                           \
        case ncclUint8: {                           \
          __res = func<uint8_t>(__VA_ARGS__);       \
          break;                                    \
        }                                           \
        case ncclInt32: {                           \
          __res = func<int32_t>(__VA_ARGS__);       \
          break;                                    \
        }                                           \
        case ncclUint32: {                          \
          __res = func<uint32_t>(__VA_ARGS__);      \
          break;                                    \
        }                                           \
        case ncclInt64: {                           \
          __res = func<int64_t>(__VA_ARGS__);       \
          break;                                    \
        }                                           \
        case ncclUint64: {                          \
          __res = func<uint64_t>(__VA_ARGS__);      \
          break;                                    \
        }                                           \
        case ncclFloat16: {                         \
          __res = func<half>(__VA_ARGS__);          \
          break;                                    \
        }                                           \
        case ncclFloat32: {                         \
          __res = func<float>(__VA_ARGS__);         \
          break;                                    \
        }                                           \
        case ncclFloat64: {                         \
          __res = func<double>(__VA_ARGS__);        \
          break;                                    \
        }                                           \
        case ncclBfloat16: {                        \
          __res = func<__nv_bfloat16>(__VA_ARGS__); \
          break;                                    \
        }                                           \
        default: {                                  \
          __res = ncclInvalidArgument;              \
        }                                           \
      }                                             \
    } while (0);                                    \
    __res;                                          \
  })
#else
#define NCCL_TYPED_CALL(ncclDataType, func, ...) \
  ({                                             \
    ncclResult_t __res;                          \
    do {                                         \
      switch (ncclDataType) {                    \
        case ncclInt8: {                         \
          __res = func<char>(__VA_ARGS__);       \
          break;                                 \
        }                                        \
        case ncclUint8: {                        \
          __res = func<uint8_t>(__VA_ARGS__);    \
          break;                                 \
        }                                        \
        case ncclInt32: {                        \
          __res = func<int32_t>(__VA_ARGS__);    \
          break;                                 \
        }                                        \
        case ncclUint32: {                       \
          __res = func<uint32_t>(__VA_ARGS__);   \
          break;                                 \
        }                                        \
        case ncclInt64: {                        \
          __res = func<int64_t>(__VA_ARGS__);    \
          break;                                 \
        }                                        \
        case ncclUint64: {                       \
          __res = func<uint64_t>(__VA_ARGS__);   \
          break;                                 \
        }                                        \
        case ncclFloat16: {                      \
          __res = func<half>(__VA_ARGS__);       \
          break;                                 \
        }                                        \
        case ncclFloat32: {                      \
          __res = func<float>(__VA_ARGS__);      \
          break;                                 \
        }                                        \
        case ncclFloat64: {                      \
          __res = func<double>(__VA_ARGS__);     \
          break;                                 \
        }                                        \
        default: {                               \
          __res = ncclInvalidArgument;           \
        }                                        \
      }                                          \
    } while (0);                                 \
    __res;                                       \
  })
#endif

// determine the optimal grid/block size to launch kernel func
std::pair<dim3, dim3> getGridAndBlockDims(
    const void* func,
    size_t count,
    ncclDataType_t datatype,
    size_t maxBlocks);

} // namespace algorithms
} // namespace nccl
