
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <thread>

#include <fmt/format.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "AllReduceDdaNvsFlatThreadedAlgo.h"
#include "AllReduceDdaNvsTreeThreadedAlgo.h"

namespace nccl {
namespace algorithms {

TEST(AllReduceDdaNvsFlatThreadedAlgoTest, Create) {
  auto algo = std::make_unique<AllReduceDdaNvsFlatThreadedAlgo>();
  algo->allReduce();
}

TEST(AllReduceDdaNvsTreeThreadedAlgoTest, Create) {
  auto algo = std::make_unique<AllReduceDdaNvsTreeThreadedAlgo>();
  algo->allReduce();
}

} // namespace algorithms
} // namespace nccl
