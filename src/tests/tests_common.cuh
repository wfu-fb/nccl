// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef TESTS_COMMON_CUH_
#define TESTS_COMMON_CUH_

#include "cuda.h"
#include "mpi.h"
#include <tuple>

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

#define MPICHECK_TEST(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK_TEST(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK_TEST(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

std::tuple<int, int, int, ncclComm_t> setupNccl(int argc, char **argv) {
  int localRank, globalRank, numRanks = 0;
  ncclUniqueId id;
  ncclComm_t comm;

  //initializing MPI
  MPICHECK_TEST(MPI_Init(&argc, &argv));
  MPICHECK_TEST(MPI_Comm_rank(MPI_COMM_WORLD, &globalRank));
  MPICHECK_TEST(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[numRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[globalRank] = getHostHash(hostname);
  MPICHECK_TEST(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<numRanks; p++) {
     if (p == globalRank) break;
     if (hostHashs[p] == hostHashs[globalRank]) localRank++;
  }

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (globalRank == 0) ncclGetUniqueId(&id);
  MPICHECK_TEST(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK_TEST(cudaSetDevice(localRank));

  //initializing NCCL
  NCCLCHECK_TEST(ncclCommInitRank(&comm, numRanks, id, globalRank));

  return std::make_tuple(localRank, globalRank, numRanks, comm);
}

void cleanupNccl(ncclComm_t globalComm) {
  //finalizing NCCL
  ncclCommDestroy(globalComm);

  //finalizing MPI
  MPICHECK_TEST(MPI_Finalize());
}

#endif
