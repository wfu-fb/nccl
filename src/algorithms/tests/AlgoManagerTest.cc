// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <thread>

#include <fmt/format.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "AlgoManager.h"
#include "checks.h"
#include "comm.h"
#include "nccl.h"

namespace nccl {
namespace algorithms {

TEST(AlgoManagerTest, Create) {
  ncclUniqueId commId;
  cudaStream_t stream;
  ncclComm_t comm;
  const size_t count = 1024;
  const int nRanks = 1;
  void* sendbuf_d{nullptr};
  void* recvbuf_d{nullptr};

  NCCLCHECKIGNORE(ncclGetUniqueId(&commId));
  CUDACHECKIGNORE(cudaStreamCreate(&stream));
  NCCLCHECKIGNORE(ncclCommInitRank(&comm, nRanks, commId, 0));

  CUDACHECKIGNORE(cudaSetDevice(0));
  CUDACHECKIGNORE(cudaMalloc(&sendbuf_d, count * sizeof(float)));
  CUDACHECKIGNORE(cudaMalloc(&recvbuf_d, count * sizeof(float)));

  EXPECT_TRUE(comm->algoMgr);
  auto algo = comm->algoMgr->getAllReduceAlgo(
      sendbuf_d, recvbuf_d, count, ncclFloat, ncclSum, comm, stream);
  EXPECT_EQ(algo, nullptr);
}

} // namespace algorithms
} // namespace nccl
