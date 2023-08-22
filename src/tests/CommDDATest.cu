// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include <gtest/gtest.h>
#include <nccl.h>
#include "../include/comm_dda.h"

TEST(CommDDATest, Clique) {
  CUDACHECKIGNORE(cudaSetDevice(0));

  std::vector<int> gpuClique{0, 1};
  auto clique = std::make_unique<ddaClique>(gpuClique);
  clique->insertRank(0, 0);

  EXPECT_EQ(clique->gpus.size(), 2);
  EXPECT_NE(clique->barrierMbox[0], nullptr);
  EXPECT_NE(clique->barrierMbox[1], nullptr);
}
