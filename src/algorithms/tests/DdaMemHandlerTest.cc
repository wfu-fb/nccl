// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <thread>

#include <fmt/format.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "DdaMemHandler.h"
#include "DdaThreadedData.h"
#include "checks.h"
#include "comm.h"

namespace nccl {
namespace algorithms {

TEST(DdaMemHandler, ThreadedRanks) {
  ncclUniqueId commId;
  NCCLCHECKIGNORE(ncclGetUniqueId(&commId));

  // allocate dev memory on each rank
  void* rank0_addr0 = nullptr;
  void* rank0_addr1 = nullptr;
  void* rank1_addr0 = nullptr;
  void* rank1_addr1 = nullptr;
  CUDACHECKIGNORE(cudaSetDevice(0));
  CUDACHECKIGNORE(cudaMalloc(&rank0_addr0, 16));
  CUDACHECKIGNORE(cudaMalloc(&rank0_addr1, 16));
  CUDACHECKIGNORE(cudaSetDevice(1));
  CUDACHECKIGNORE(cudaMalloc(&rank1_addr0, 16));
  CUDACHECKIGNORE(cudaMalloc(&rank1_addr1, 16));

  // test helper function
  auto tester = [&](int rank) {
    const int nRanks = 2;
    CUDACHECKIGNORE(cudaSetDevice(rank));
    ncclComm_t comm;
    NCCLCHECKIGNORE(ncclCommInitRank(&comm, nRanks, commId, rank));

    // add local dev addresses
    DdaMemHandler handler(comm);
    if (rank == 0) {
      handler.add(rank0_addr0);
      handler.add(rank0_addr1);
    } else {
      handler.add(rank1_addr0);
      handler.add(rank1_addr1);
    }
    NCCLCHECKIGNORE(handler.exchangeMemHandles());
    VLOG(1) << "rank " << rank << ": exchangeMemHandles done.";

    // verify memory addresses
    EXPECT_EQ(handler.get(0, 0), rank0_addr0);
    EXPECT_EQ(handler.get(0, 1), rank0_addr1);
    EXPECT_EQ(handler.get(0, 2), nullptr);
    EXPECT_EQ(handler.get(1, 0), rank1_addr0);
    EXPECT_EQ(handler.get(1, 1), rank1_addr1);
    EXPECT_EQ(handler.get(1, 2), nullptr);
    VLOG(1) << "rank " << rank << ": verified memory addresses.";
  };

  auto t0 = std::thread([&] { tester(0); });
  auto t1 = std::thread([&] { tester(1); });

  t0.join();
  t1.join();
}

} // namespace algorithms
} // namespace nccl
