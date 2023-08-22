// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <nccl.h>
#include <iostream>
#include <thread>
#include "../include/checks.h"
#include "../include/collectives.h"

/**
 * GpuAgent manages one given GPU device
 * each collective() call is expeceted to be invoked
 * on a per-device thread. see NVSFlat as examples
 */
class GpuAgent {
 public:
  GpuAgent(int rank, int totalRank, int count, uintptr_t* barrierMbox_d)
      : rank_(rank),
        totalRank_(totalRank),
        count_(count),
        barrierMbox_d_(barrierMbox_d) {}

  ~GpuAgent() {
    if (not initialized_) {
      return;
    }
    CUDACHECKIGNORE(cudaSetDevice(rank_));
    CUDACHECKIGNORE(cudaFree(sendbuf_d_));
    CUDACHECKIGNORE(cudaFree(recvbuf_d_));
    CUDACHECKIGNORE(cudaFree(tmpbuf_d_));
    free(sendbuf_);
    free(recvbuf_);
    free(expbuf_);
  }

  void init() {
    const size_t size = count_ * sizeof(float);
    CUDACHECKIGNORE(cudaSetDevice(rank_));
    CUDACHECKIGNORE(cudaMalloc(&sendbuf_d_, size));
    CUDACHECKIGNORE(cudaMalloc(&recvbuf_d_, size));
    CUDACHECKIGNORE(cudaMalloc(&tmpbuf_d_, size));

    sendbuf_ = static_cast<float*>(malloc(size));
    recvbuf_ = static_cast<float*>(malloc(size));
    expbuf_ = static_cast<float*>(malloc(size));

    for (int i = 0; i < count_; ++i) {
      sendbuf_[i] = i + rank_ * count_;

      expbuf_[i] = 0.0f;
      for (int rankId = 0; rankId < totalRank_; ++rankId) {
        expbuf_[i] += i + rankId * count_;
      }
    }

    CUDACHECKIGNORE(cudaMemcpy(sendbuf_d_, sendbuf_, size, cudaMemcpyDefault));

    // assume NVS full-mesh architecture
    for (int peerRank = 0; peerRank < totalRank_; ++peerRank) {
      if (rank_ == peerRank) {
        continue;
      }

      CUDACHECKIGNORE(cudaDeviceEnablePeerAccess(peerRank, 0));
    }

    initialized_ = true;
  }

  void verifyRecvbuf() {
    const size_t size = count_ * sizeof(float);
    CUDACHECKIGNORE(cudaMemcpy(recvbuf_, recvbuf_d_, size, cudaMemcpyDefault));
    for (int i = 0; i < count_; ++i) {
      EXPECT_EQ(recvbuf_[i], expbuf_[i]);
    }
  }

  void runAllreduceFlatAndVerify() {
    assert(initialized_);
    CUDACHECKIGNORE(cudaSetDevice(rank_));
    uintptr_t barrierFlag = 1;
    ncclKernel_AllReduce_DDA_Flat<float, 2><<<1, 2>>>(
        barrierMbox_d_, barrierFlag, rank_, sendbuf_d_, recvbuf_d_, count_);

    CUDACHECKIGNORE(cudaDeviceSynchronize());

    verifyRecvbuf();
  }

  void runAllreduceTreeAndVerify() {
    assert(initialized_);
    CUDACHECKIGNORE(cudaSetDevice(rank_));
    uintptr_t barrierFlag = 1;
    ncclKernel_AllReduce_DDA_Tree<float, 2><<<1, 2>>>(
        barrierMbox_d_,
        barrierFlag,
        rank_,
        sendbuf_d_,
        tmpbuf_d_,
        recvbuf_d_,
        count_);

    CUDACHECKIGNORE(cudaDeviceSynchronize());

    verifyRecvbuf();
  }

 private:
  const int rank_{0};
  const int totalRank_{0};
  const int count_{0};
  bool initialized_{false};

  float* sendbuf_{nullptr};
  float* recvbuf_{nullptr};
  float* expbuf_{nullptr};

  // device bufs
  float* sendbuf_d_{nullptr};
  float* recvbuf_d_{nullptr};
  float* tmpbuf_d_{nullptr};
  uintptr_t* barrierMbox_d_{nullptr};
};

TEST(AllreduceDDATest, NVSFlat) {
  const int totalRank{2};

  // init global barrier
  const size_t barrierSize = 2 * totalRank * totalRank * sizeof(uintptr_t);
  uintptr_t* barrier_d;
  CUDACHECKIGNORE(cudaSetDevice(0));
  CUDACHECKIGNORE(cudaMalloc(&barrier_d, barrierSize));
  CUDACHECKIGNORE(cudaMemset(barrier_d, 0, barrierSize));

  auto agent0 = std::make_unique<GpuAgent>(0, totalRank, 32, barrier_d);
  auto agent1 = std::make_unique<GpuAgent>(1, totalRank, 32, barrier_d);

  agent0->init();
  agent1->init();

  auto t0 = std::thread([&agent0] { agent0->runAllreduceFlatAndVerify(); });
  auto t1 = std::thread([&agent1] { agent1->runAllreduceFlatAndVerify(); });
  t0.join();
  t1.join();

  agent0.reset();
  agent1.reset();
}

TEST(AllreduceDDATest, NVSTree) {
  const int totalRank{2};

  // init global barrier
  const size_t barrierSize = 2 * totalRank * totalRank * sizeof(uintptr_t);
  uintptr_t* barrier_d;
  CUDACHECKIGNORE(cudaSetDevice(0));
  CUDACHECKIGNORE(cudaMalloc(&barrier_d, barrierSize));
  CUDACHECKIGNORE(cudaMemset(barrier_d, 0, barrierSize));

  auto agent0 = std::make_unique<GpuAgent>(0, totalRank, 32, barrier_d);
  auto agent1 = std::make_unique<GpuAgent>(1, totalRank, 32, barrier_d);

  agent0->init();
  agent1->init();

  auto t1 = std::thread([&agent1] { agent1->runAllreduceTreeAndVerify(); });
  auto t0 = std::thread([&agent0] { agent0->runAllreduceTreeAndVerify(); });
  t0.join();
  t1.join();

  agent0.reset();
  agent1.reset();
}
