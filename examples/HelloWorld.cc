// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <utils.h>
#include "nccl.h"
#include "cudawrapper.h"

int main(int argc, char* argv[]) {
  int rank = 0, nproc = 1, localRank = 0;
  int count = 1024 * 1024;

  ncclComm_t comm;
  cudaStream_t stream;
  int* userBuff = NULL;
  CudaWrapper* cudaWrapper = ncclSetupWrappers(false);

  ncclUniqueId ncclId;
  NCCLCHECK(ncclGetUniqueId(&ncclId));

  printf("Hello world. NCCL_VERSION %d-%s\n", NCCL_VERSION_CODE, NCCL_SUFFIX);

  CUDACHECK(cudaWrapper->cudaSetDevice(localRank));
  CUDACHECK(cudaWrapper->cudaStreamCreate(&stream));
  NCCLCHECK(ncclCommInitRank(&comm, nproc, ncclId, rank));

  CUDACHECK(cudaWrapper->cudaMalloc((void**)&userBuff, count * sizeof(int)));
  NCCLCHECK(ncclAllReduce(
      (const void*)userBuff, userBuff, count, ncclInt, ncclSum, comm, stream));

  CUDACHECK(cudaWrapper->cudaFree(userBuff));
  CUDACHECK(cudaWrapper->cudaSetDevice(localRank));
  CUDACHECK(cudaWrapper->cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommDestroy(comm));

  return EXIT_SUCCESS;
}
