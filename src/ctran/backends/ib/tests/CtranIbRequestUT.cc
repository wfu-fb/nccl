// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include "../CtranIb.h"
#include "checks.h"

class CtranIbRequestTest : public ::testing::Test {
 public:
  CtranIbRequestTest() = default;
};

TEST_F(CtranIbRequestTest, Complete) {
  CtranIbRequest req;
  req.complete();
  EXPECT_TRUE(req.isComplete());
}

TEST_F(CtranIbRequestTest, SetRefCount) {
  CtranIbRequest req;

  req.setRefCount(3);
  req.complete(); // refCount reduced to 2
  EXPECT_FALSE(req.isComplete());

  req.complete(); // refCount reduced to 1
  req.complete(); // refCount reduced to 0, complete
  EXPECT_TRUE(req.isComplete());
}
