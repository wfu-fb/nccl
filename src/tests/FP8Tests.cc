// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <memory>
#include "checks.h"
#include "comm.h"
#include "core.h"

class FP8Test : public ::testing::Test {
 public:
  FP8Test() = default;
};

#if defined(NCCL_ENABLE_FP8)
TEST_F(FP8Test, ncclFp8E4M3) {
  ncclDataType_t type = ncclFp8E4M3;
  size_t nbytes = ncclTypeSize(type);
  EXPECT_EQ(nbytes, 1);
}

TEST_F(FP8Test, ncclFp8E5M2) {
  ncclDataType_t type = ncclFp8E5M2;
  size_t nbytes = ncclTypeSize(type);
  EXPECT_EQ(nbytes, 1);
}
#endif
