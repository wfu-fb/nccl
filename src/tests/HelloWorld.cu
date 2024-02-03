#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include "tests_common.cuh"
#include "cudawrapper.h"

int main(int argc, char* argv[])
{
  int size = 32*1024*1024;

  int localRank, globalRank, numRanks;

  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;
  CudaWrapper* cudaWrapper_ = ncclSetupWrappers(false);

  std::tie(localRank, globalRank, numRanks, comm) = setupNccl(argc, argv);

  CUDACHECK_TEST(
      cudaWrapper->cudaMalloc((void**)&sendbuff, size * sizeof(float)));
  CUDACHECK_TEST(
      cudaWrapper->cudaMalloc((void**)&recvbuff, size * sizeof(float)));
  CUDACHECK_TEST(cudaWrapper->cudaStreamCreate(&s));

  //communicating using NCCL
  NCCLCHECK_TEST(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
        comm, s));

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK_TEST(cudaWrapper->cudaStreamSynchronize(s));

  //free device buffers
  CUDACHECK_TEST(cudaWrapper->cudaFree(sendbuff));
  CUDACHECK_TEST(cudaWrapper->cudaFree(recvbuff));

  cleanupNccl(comm);

  printf("[MPI Rank %d] Success \n", globalRank);
  return 0;
}
