// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <utils.h>

int main(int argc, char* argv[]) {
  int rank = 0, nproc = 1, localRank = 0;
  int count = 1024 * 1024;

  ncclComm_t comm;
  cudaStream_t stream;
  int* userBuff = NULL;
  void* userBuffHandle = NULL;

  ncclUniqueId ncclId;
  NCCLCHECK(ncclGetUniqueId(&ncclId));

  printf(
      "NCCL register API sanity check. NCCL_VERSION %d-%s\n",
      NCCL_VERSION_CODE,
      NCCL_SUFFIX);

  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaStreamCreate(&stream));
  NCCLCHECK(ncclCommInitRank(&comm, nproc, ncclId, rank));

  CUDACHECK(cudaMalloc(&userBuff, count * sizeof(int)));

  NCCLCHECK(
      ncclCommRegister(comm, userBuff, count * sizeof(int), &userBuffHandle));

  NCCLCHECK(ncclCommDeregister(comm, userBuffHandle));

  CUDACHECK(cudaFree(userBuff));
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommDestroy(comm));

  return EXIT_SUCCESS;
}
