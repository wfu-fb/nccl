// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <thread>

#include <fmt/format.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "AlgoDirector.h"
#include "AlgoInit.h"
#include "checks.h"
#include "comm.h"
#include "nccl.h"

namespace nccl {
namespace algorithms {

TEST(AlgoDirectorTest, Create) {
  ncclUniqueId commId;
  cudaStream_t stream;
  ncclComm_t comm;
  const size_t count = 1024;
  const int nRanks = 1;
  void* sendbuff_d{nullptr};
  void* recvbuff_d{nullptr};

  NCCLCHECKIGNORE(ncclGetUniqueId(&commId));
  CUDACHECKIGNORE(cudaStreamCreate(&stream));
  NCCLCHECKIGNORE(ncclCommInitRank(&comm, nRanks, commId, 0));
  NCCLCHECKIGNORE(algoInit(comm, true));

  CUDACHECKIGNORE(cudaSetDevice(0));
  CUDACHECKIGNORE(cudaMalloc(&sendbuff_d, count * sizeof(float)));
  CUDACHECKIGNORE(cudaMalloc(&recvbuff_d, count * sizeof(float)));

  EXPECT_TRUE(comm->algoDirector);
  EXPECT_TRUE(comm->algoDirector->allReduce);
  auto algo = comm->algoDirector->allReduce->getAlgoAllReduce(
      sendbuff_d, recvbuff_d, count, ncclFloat, ncclSum, comm, stream);
  EXPECT_EQ(algo, nullptr);

  CUDACHECKIGNORE(cudaFree(sendbuff_d));
  CUDACHECKIGNORE(cudaFree(recvbuff_d));
  CUDACHECKIGNORE(cudaStreamDestroy(stream));
  NCCLCHECKIGNORE(ncclCommDestroy(comm));
}

TEST(AlgoDirectorTest, CanRunDdaThreaded) {
  void* sendbuff_d{nullptr};
  void* recvbuff_d{nullptr};
  CUDACHECKIGNORE(cudaMalloc(&sendbuff_d, 1024));
  CUDACHECKIGNORE(cudaMalloc(&recvbuff_d, 1024));

  ncclComm comm;
  size_t totalBytes;
  size_t numDdaThreads;
  const size_t kTreeThresholdBytes = 2048;

  {
    // not all threaded ranks
    comm.nRanks = 8;
    numDdaThreads = 4;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceThreaded(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      numDdaThreads,
      kTreeThresholdBytes);
    EXPECT_FALSE(ret);
  }

  {
    // single rank
    comm.nRanks = 1;
    numDdaThreads = 1;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceThreaded(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      numDdaThreads,
      kTreeThresholdBytes);
    EXPECT_FALSE(ret);
  }

  {
    // not power of 2 ranks
    comm.nRanks = 6;
    numDdaThreads = 6;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceThreaded(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      numDdaThreads,
      kTreeThresholdBytes);
    EXPECT_FALSE(ret);
  }

  {
    // more than 8 ranks
    comm.nRanks = 16;
    numDdaThreads = 16;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceThreaded(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      numDdaThreads,
      kTreeThresholdBytes);
    EXPECT_FALSE(ret);
  }

  {
    // not ncclSum op
    comm.nRanks = 8;
    numDdaThreads = 8;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceThreaded(
      &comm,
      ncclMax,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      numDdaThreads,
      kTreeThresholdBytes);
    EXPECT_FALSE(ret);
  }

  {
    // flat in-place (not supported yet)
    comm.nRanks = 8;
    numDdaThreads = 8;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceThreaded(
      &comm,
      ncclSum,
      sendbuff_d,
      sendbuff_d,
      totalBytes,
      numDdaThreads,
      kTreeThresholdBytes);
    EXPECT_FALSE(ret);
  }

  {
    // flat: totalBytes not multiple of 16
    comm.nRanks = 8;
    numDdaThreads = 8;
    totalBytes = 1032;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceThreaded(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      numDdaThreads,
      kTreeThresholdBytes);
    EXPECT_FALSE(ret);
  }

  {
    // tree: totalBytes not multiple of 16 * nRanks
    comm.nRanks = 8;
    numDdaThreads = 8;

    // > 2048 tree threshold, but not multiple of 128
    totalBytes = 2064;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceThreaded(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      numDdaThreads,
      kTreeThresholdBytes);
    EXPECT_FALSE(ret);
  }

  {
    // flat: valid case
    comm.nRanks = 8;
    numDdaThreads = 8;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceThreaded(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      numDdaThreads,
      kTreeThresholdBytes);
    EXPECT_TRUE(ret);
  }

  {
    // tree: valid case
    comm.nRanks = 8;
    numDdaThreads = 8;

    // > 2048 tree threshold, and multiple of 128
    totalBytes = 2176;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceThreaded(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      numDdaThreads,
      kTreeThresholdBytes);
    EXPECT_TRUE(ret);
  }

  CUDACHECKIGNORE(cudaFree(sendbuff_d));
  CUDACHECKIGNORE(cudaFree(recvbuff_d));
}

TEST(AlgoDirectorTest, CanRunDdaIpc) {
  void* sendbuff_d{nullptr};
  void* recvbuff_d{nullptr};
  CUDACHECKIGNORE(cudaMalloc(&sendbuff_d, 1024));
  CUDACHECKIGNORE(cudaMalloc(&recvbuff_d, 1024));

  ncclComm comm;
  size_t totalBytes;
  const size_t kTreeThresholdBytes = 2048;
  const size_t kTmpbuffSize = 32 * 1024;

  {
    // not all local ranks
    comm.nRanks = 8;
    comm.localRanks = 4;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceIpc(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      kTreeThresholdBytes,
      kTmpbuffSize);
    EXPECT_FALSE(ret);
  }

  {
    // single rank
    comm.nRanks = 1;
    comm.localRanks = 1;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceIpc(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      kTreeThresholdBytes,
      kTmpbuffSize);
    EXPECT_FALSE(ret);
  }

  {
    // not power of 2 ranks
    comm.nRanks = 6;
    comm.localRanks = 6;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceIpc(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      kTreeThresholdBytes,
      kTmpbuffSize);
    EXPECT_FALSE(ret);
  }

  {
    // not ncclSum op
    comm.nRanks = 8;
    comm.localRanks = 8;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceIpc(
      &comm,
      ncclMax,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      kTreeThresholdBytes,
      kTmpbuffSize);
    EXPECT_FALSE(ret);
  }

  {
    // flat: totalBytes not multiple of 16
    comm.nRanks = 8;
    comm.localRanks = 8;
    totalBytes = 1032;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceIpc(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      kTreeThresholdBytes,
      kTmpbuffSize);
    EXPECT_FALSE(ret);
  }

  {
    // tree: totalBytes not multiple of 16 * nRanks
    comm.nRanks = 8;
    comm.localRanks = 8;

    // > 2048 tree threshold, but not multiple of 128
    totalBytes = 2064;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceIpc(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      kTreeThresholdBytes,
      kTmpbuffSize);
    EXPECT_FALSE(ret);
  }

  {
    // totalBytes > tmpBuferSize
    comm.nRanks = 8;
    comm.localRanks = 8;
    totalBytes = 1024 * 1024 * 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceIpc(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      kTreeThresholdBytes,
      kTmpbuffSize);
    EXPECT_FALSE(ret);
  }

  {
    // flat: valid case
    comm.nRanks = 8;
    comm.localRanks = 8;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceIpc(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      kTreeThresholdBytes,
      kTmpbuffSize);
    EXPECT_TRUE(ret);
  }

  {
    // tree: valid case
    comm.nRanks = 8;
    comm.localRanks = 8;

    // > 2048 tree threshold, and multiple of 128
    totalBytes = 2176;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceIpc(
      &comm,
      ncclSum,
      sendbuff_d,
      recvbuff_d,
      totalBytes,
      kTreeThresholdBytes,
      kTmpbuffSize);
    EXPECT_TRUE(ret);
  }

  {
    // flat in-place (supported in IPC)
    comm.nRanks = 8;
    comm.localRanks = 8;
    totalBytes = 1024;
    auto ret = AlgoManagerAllReduce::canRunDdaAllReduceIpc(
      &comm,
      ncclSum,
      sendbuff_d,
      sendbuff_d,
      totalBytes,
      kTreeThresholdBytes,
      kTmpbuffSize);
    EXPECT_TRUE(ret);
  }

  CUDACHECKIGNORE(cudaFree(sendbuff_d));
  CUDACHECKIGNORE(cudaFree(recvbuff_d));
}

} // namespace algorithms
} // namespace nccl
