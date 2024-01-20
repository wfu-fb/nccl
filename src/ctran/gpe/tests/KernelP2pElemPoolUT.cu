// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include "CtranGpeDev.h"
#include "CtranGpeImpl.h"
#include "tests_common.cuh"

class KernelP2pElemPoolTest : public ::testing::Test {
 public:
  int cudaDev;
  KernelP2pElemPoolTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    CUDACHECK_TEST(cudaSetDevice(cudaDev));
  }
};

TEST_F(KernelP2pElemPoolTest, Initialize) {
  constexpr int poolSize = 1000;
  auto elemPool =
      std::unique_ptr<KernelP2pElemPool>(new KernelP2pElemPool(poolSize));

  ASSERT_NE(elemPool, nullptr);
  EXPECT_EQ(elemPool->size(), poolSize);
}

TEST_F(KernelP2pElemPoolTest, InvalidNGroup) {
  constexpr int poolSize = 1000;
  auto elemPool =
      std::unique_ptr<KernelP2pElemPool>(new KernelP2pElemPool(poolSize));

  ASSERT_NE(elemPool, nullptr);
  EXPECT_EQ(elemPool->size(), poolSize);

  auto elem = elemPool->pop(CTRAN_ALGO_MAX_THREAD_BLOCKS+1);
  EXPECT_EQ(elem, nullptr);
}

__global__ void P2pElemConsumerKernel(KernelP2pElem* elemList) {
  KernelP2pElem* elem = elemList;
  while (elem) {
    elem->inuse[blockIdx.x] = false;
    elem = elem->next;
  }
}

TEST_F(KernelP2pElemPoolTest, PopReclaim) {
  constexpr int nElems = 5;
  constexpr int poolSize = 1000;
  auto elemPool =
      std::unique_ptr<KernelP2pElemPool>(new KernelP2pElemPool(poolSize));
  ASSERT_NE(elemPool, nullptr);

  constexpr int ngroups = 5;

  // Pop some elements from freePool, stored as C-style list for kernel to access
  KernelP2pElem *prevElem = nullptr, *elemList = nullptr;
  for (int i = 0; i < nElems; i++) {
    auto elem = elemPool->pop(ngroups);
    if (prevElem) {
      prevElem->next = elem; // append to existing list
    } else {
      elemList = elem; // head of list
    }
    prevElem = elem;

    // Expect each has been marked as inuse
    std::vector<int> inuse(elem->inuse, elem->inuse + ngroups);
    EXPECT_THAT(inuse, testing::Each(true));
  }

  // Check current size of freePool
  EXPECT_EQ(elemPool->size(), poolSize - nElems);

  // Launch kernel to consume these elements, with ngroups gridSize
  dim3 grid = {ngroups, 1, 1};
  dim3 blocks = {1, 1, 1};
  void* args[] = {&elemList};
  CUDACHECK_TEST(cudaLaunchKernel(
      reinterpret_cast<void*>(P2pElemConsumerKernel),
      grid,
      blocks,
      args,
      0,
      0));
  CUDACHECK_TEST(cudaStreamSynchronize(0));

  // Reclaim no longer inuse elements, and check pool size has increased back
  elemPool->reclaim();
  EXPECT_EQ(elemPool->size(), poolSize);
}
