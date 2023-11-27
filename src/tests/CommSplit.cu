// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <assert.h>
#include <nccl.h>
#include <iostream>
#include <thread>
#include <vector>

#include "mpi.h"
#include "tests_common.cuh"

int main(int argc, char* argv[]) {
  char* progname = argv[0];
  int rank, nRanks;
  MPI_Comm localComm;
  int localRank;
  cudaStream_t s;
  ncclComm_t comm, splitComm, connectComm;
  void *sendBuf, *recvBuf, *tmpRecvBuf;
  int *hostBuf;
  int errors = 0;
  constexpr int count = (1024 * 1024);

  std::tie(localRank, rank, nRanks, comm) = setupNccl(argc, argv);

  CUDACHECK_TEST(cudaSetDevice(localRank));
  CUDACHECK_TEST(cudaStreamCreate(&s));

  CUDACHECK_TEST(cudaHostAlloc(&hostBuf, count * sizeof(int), 0));
  for (int i = 0; i < count; i++) {
    hostBuf[i] = rank;
  }

  CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
  CUDACHECK_TEST(cudaMalloc(&tmpRecvBuf, count * sizeof(int)));

  /* Perform a 2D split of the ranks.  The first dimension would
   * contain a subset of ranks (half the ranks if we use CommSplit and
   * the ranks on each node if we use commSplitType).  The second
   * dimension would connect corresponding ranks from each of these split
   * communicators.  Doing an AllReduce on both dimensions would
   * result in the same value as doing an AllReduce on the global
   * communicator. */
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,18,3)
#if defined(ENABLE_COMM_SPLIT_TYPE) && defined(NCCL_COMM_SPLIT_TYPE_SUPPORTED)
  NCCLCHECK_TEST(ncclCommSplitType(comm, NCCL_SPLIT_TYPE_NODE, 0, &splitComm, nullptr));
#else
  NCCLCHECK_TEST(ncclCommSplit(comm, rank / 2, 0, &splitComm, nullptr));
#endif
  NCCLCHECK_TEST(ncclCommUserRank(splitComm, &localRank));

  NCCLCHECK_TEST(ncclCommSplit(comm, localRank, 0, &connectComm, nullptr));

  CUDACHECK_TEST(cudaMemcpyAsync(sendBuf, hostBuf, count * sizeof(int),
        cudaMemcpyDefault, s));
  NCCLCHECK_TEST(ncclAllReduce(sendBuf, tmpRecvBuf, count, ncclInt,
        ncclSum, splitComm, s));
  NCCLCHECK_TEST(ncclAllReduce(tmpRecvBuf, recvBuf, count, ncclInt,
        ncclSum, connectComm, s));
  CUDACHECK_TEST(cudaMemcpyAsync(hostBuf, recvBuf, count * sizeof(int),
        cudaMemcpyDefault, s));

  CUDACHECK_TEST(cudaStreamSynchronize(s));

  for (int i = 0; i < count; i++) {
    if (hostBuf[i] != nRanks * (nRanks - 1) / 2) {
      printf("ERROR: hostBuf[%d]=%d, expected %d\n", i, hostBuf[i],
          nRanks * (nRanks - 1) / 2);
      errors++;
      break;
    }
  }
#endif

  CUDACHECK_TEST(cudaStreamDestroy(s));

  CUDACHECK_TEST(cudaFree(tmpRecvBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFreeHost(hostBuf));

  NCCLCHECK_TEST(ncclCommDestroy(connectComm));
  NCCLCHECK_TEST(ncclCommDestroy(splitComm));

  MPICHECK_TEST(MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INT, MPI_SUM,
      MPI_COMM_WORLD));
  cleanupNccl(comm);

  if (errors == 0 && rank == 0) {
    printf("No Errors found!\n");
  }

  return 0;
}
