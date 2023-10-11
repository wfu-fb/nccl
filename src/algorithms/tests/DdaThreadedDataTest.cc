
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <thread>

#include <fmt/format.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "DdaThreadedData.h"

namespace nccl {
namespace algorithms {

TEST(DdaThreadedDataTest, Get) {
  DdaThreadedData* data0;
  DdaThreadedData* data1;
  auto t0 = std::thread([&data0] { data0 = DdaThreadedData::get(); });
  auto t1 = std::thread([&data1] { data1 = DdaThreadedData::get(); });
  t0.join();
  t1.join();

  EXPECT_EQ(data0, data1);
  DdaThreadedData::get()->clear();
}

TEST(DdaThreadedDataTest, RegisterUnregister) {
  const uint64_t commHash = 123;

  auto t0 = std::thread([&commHash] {
    EXPECT_TRUE(DdaThreadedData::get()->registerRank(commHash, 0));
  });
  auto t1 = std::thread([&commHash] {
    EXPECT_TRUE(DdaThreadedData::get()->registerRank(commHash, 1));
  });
  t0.join();
  t1.join();

  EXPECT_TRUE(DdaThreadedData::get()->hasRank(commHash, 0));
  EXPECT_TRUE(DdaThreadedData::get()->hasRank(commHash, 1));
  EXPECT_FALSE(DdaThreadedData::get()->hasRank(commHash, 2));

  auto t2 = std::thread([&commHash] {
    EXPECT_TRUE(DdaThreadedData::get()->unregisterRank(commHash, 0));
  });
  t2.join();

  EXPECT_FALSE(DdaThreadedData::get()->hasRank(commHash, 0));
  DdaThreadedData::get()->clear();
}

TEST(DdaThreadedDataTest, RegisterUnregisterMultiComm) {
  const uint64_t commHash0 = 123;
  const uint64_t commHash1 = 456;

  auto t0 = std::thread([&commHash0, &commHash1] {
    EXPECT_TRUE(DdaThreadedData::get()->registerRank(commHash0, 0));
    EXPECT_TRUE(DdaThreadedData::get()->registerRank(commHash1, 0));
  });
  auto t1 = std::thread([&commHash0, &commHash1] {
    EXPECT_TRUE(DdaThreadedData::get()->registerRank(commHash0, 1));
    EXPECT_TRUE(DdaThreadedData::get()->registerRank(commHash1, 1));
  });
  t0.join();
  t1.join();

  EXPECT_TRUE(DdaThreadedData::get()->hasRank(commHash0, 0));
  EXPECT_TRUE(DdaThreadedData::get()->hasRank(commHash0, 1));
  EXPECT_FALSE(DdaThreadedData::get()->hasRank(commHash0, 2));

  EXPECT_TRUE(DdaThreadedData::get()->hasRank(commHash1, 0));
  EXPECT_TRUE(DdaThreadedData::get()->hasRank(commHash1, 1));
  EXPECT_FALSE(DdaThreadedData::get()->hasRank(commHash1, 2));

  auto t2 = std::thread([&commHash1] {
    EXPECT_TRUE(DdaThreadedData::get()->unregisterRank(commHash1, 0));
  });
  t2.join();

  EXPECT_FALSE(DdaThreadedData::get()->hasRank(commHash1, 0));
  DdaThreadedData::get()->clear();
}

} // namespace algorithms
} // namespace nccl
