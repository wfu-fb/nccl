// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <thread>

#include <fmt/format.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "AllReduceDdaNvsFlatThreadedAlgo.h"
#include "AllReduceDdaNvsTreeThreadedAlgo.h"
#include "checks.h"
#include "comm.h"

namespace nccl {
namespace algorithms {

/**
 * GpuAgent manages one GPU device
 */
class GpuAgent {
 public:
  GpuAgent(
      int rank,
      int nRanks,
      int count,
      const ncclUniqueId& commId,
      int nIter)
      : rank_(rank),
        nRanks_(nRanks),
        count_(count),
        commId_(commId),
        nIter_(nIter),
        sendbufs_(nIter),
        recvbufs_(nIter),
        expbufs_(nIter),
        sendbufs_d_(nIter),
        recvbufs_d_(nIter),
        tmpbufs_d_(nIter) {}

  ~GpuAgent() {
    if (not initialized_) {
      return;
    }
    CUDACHECKIGNORE(cudaSetDevice(rank_));
    for (int i = 0; i < nIter_; ++i) {
      CUDACHECKIGNORE(cudaFree(sendbufs_d_[i]));
      CUDACHECKIGNORE(cudaFree(recvbufs_d_[i]));
      CUDACHECKIGNORE(cudaFree(tmpbufs_d_[i]));
      free(sendbufs_[i]);
      free(recvbufs_[i]);
      free(expbufs_[i]);
    }
  }

  void initBufs(int iterId) {
    assert(iterId < nIter_);
    assert(iterId >= 0);

    // init host buffs
    auto& sendbuf_d = sendbufs_d_.at(iterId);
    auto& recvbuf_d = recvbufs_d_.at(iterId);
    auto& tmpbuf_d = tmpbufs_d_.at(iterId);
    auto& sendbuf = sendbufs_.at(iterId);
    auto& recvbuf = recvbufs_.at(iterId);
    auto& expbuf = expbufs_.at(iterId);

    const size_t size = count_ * sizeof(float);
    CUDACHECKIGNORE(cudaMalloc(&sendbuf_d, size));
    CUDACHECKIGNORE(cudaMalloc(&recvbuf_d, size));
    CUDACHECKIGNORE(cudaMalloc(&tmpbuf_d, size));

    sendbuf = static_cast<float*>(malloc(size));
    recvbuf = static_cast<float*>(malloc(size));
    expbuf = static_cast<float*>(malloc(size));

    for (int i = 0; i < count_; ++i) {
      sendbuf[i] = i + rank_ * count_ + iterId;

      expbuf[i] = 0.0f;
      for (int rankId = 0; rankId < nRanks_; ++rankId) {
        expbuf[i] += i + rankId * count_ + iterId;
      }
    }

    // init dev buffs
    CUDACHECKIGNORE(cudaMemcpy(sendbuf_d, sendbuf, size, cudaMemcpyDefault));
  }

  void init() {
    // create communicator
    CUDACHECKIGNORE(cudaSetDevice(rank_));
    CUDACHECKIGNORE(cudaStreamCreate(&stream));
    NCCLCHECKIGNORE(ncclCommInitRank(&comm, nRanks_, commId_, rank_));

    // init host/dev buffs
    for (int i = 0; i < nIter_; ++i) {
      initBufs(i);
    }

    // assume NVS full-mesh architecture
    for (int peerRank = 0; peerRank < nRanks_; ++peerRank) {
      if (rank_ == peerRank) {
        continue;
      }

      CUDACHECKIGNORE(cudaDeviceEnablePeerAccess(peerRank, 0));
    }

    initialized_ = true;
  }

  void verifyRecvbuf(int iterId) {
    const size_t size = count_ * sizeof(float);

    auto& recvbuf_d = recvbufs_d_.at(iterId);
    auto& recvbuf = recvbufs_.at(iterId);
    auto& expbuf = expbufs_.at(iterId);

    CUDACHECKIGNORE(cudaMemcpy(recvbuf, recvbuf_d, size, cudaMemcpyDefault));
    for (int i = 0; i < count_; ++i) {
      EXPECT_EQ(recvbuf[i], expbuf[i]);
    }
  }

  void runAllReduceAndVerify(std::unique_ptr<AllReduceAlgo> algo, int iterId) {
    assert(initialized_);
    CUDACHECKIGNORE(cudaSetDevice(comm->cudaDev));

    auto& sendbuf_d = sendbufs_d_.at(iterId);
    auto& recvbuf_d = recvbufs_d_.at(iterId);
    NCCLCHECKIGNORE(algo->allReduce());
    CUDACHECKIGNORE(cudaDeviceSynchronize());

    verifyRecvbuf(iterId);
    VLOG(1) << fmt::format(
        "rank {}: runAllReduceAndVerify Done. IterId {}", rank_, iterId);
  }

  cudaStream_t stream;
  ncclComm_t comm;

  const int rank_{0};
  const int nRanks_{0};
  const int count_{0};
  const ncclUniqueId commId_;
  const int nIter_{0};
  bool initialized_{false};

  // host buffers
  std::vector<float*> sendbufs_{};
  std::vector<float*> recvbufs_{};
  std::vector<float*> expbufs_{};

  // dev buffers
  std::vector<float*> sendbufs_d_{};
  std::vector<float*> recvbufs_d_{};
  std::vector<float*> tmpbufs_d_{};
};

TEST(AllReduceAlgoTest, NvsFlatThreaded) {
  const int count = 16;
  const int nRanks = 2;
  const int nIter = 10;

  ncclUniqueId commId;
  NCCLCHECKIGNORE(ncclGetUniqueId(&commId));

  auto agent0 = std::make_unique<GpuAgent>(0, nRanks, count, commId, nIter);
  auto agent1 = std::make_unique<GpuAgent>(1, nRanks, count, commId, nIter);

  {
    auto t0 = std::thread([&agent0] { agent0->init(); });
    auto t1 = std::thread([&agent1] { agent1->init(); });

    t0.join();
    t1.join();
  }

  {
    auto t0 = std::thread([&] {
      for (int i = 0; i < nIter; ++i) {
        auto algo = agent0->comm->algoMgr->getAllReduceDdaNvsFlatThreadedAlgo(
            agent0->sendbufs_d_.at(i),
            agent0->recvbufs_d_.at(i),
            count,
            ncclFloat,
            ncclSum,
            agent0->comm,
            agent0->stream);
        agent0->runAllReduceAndVerify(std::move(algo), i);
      }
    });
    auto t1 = std::thread([&] {
      for (int i = 0; i < nIter; ++i) {
        auto algo = agent1->comm->algoMgr->getAllReduceDdaNvsFlatThreadedAlgo(
            agent1->sendbufs_d_.at(i),
            agent1->recvbufs_d_.at(i),
            count,
            ncclFloat,
            ncclSum,
            agent1->comm,
            agent1->stream);
        agent1->runAllReduceAndVerify(std::move(algo), i);
      }
    });

    t0.join();
    t1.join();
  }
}

TEST(AllReduceDdaNvsTreeThreadedAlgoTest, Create) {
  auto algo = std::make_unique<AllReduceDdaNvsTreeThreadedAlgo>();
  algo->allReduce();
}

} // namespace algorithms
} // namespace nccl
