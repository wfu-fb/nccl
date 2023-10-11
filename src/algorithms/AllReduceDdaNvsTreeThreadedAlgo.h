// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include "AllReduceAlgo.h"

namespace nccl {
namespace algorithms {

class AllReduceDdaNvsTreeThreadedAlgo : public AllReduceAlgo {
 public:
  AllReduceDdaNvsTreeThreadedAlgo();
  ~AllReduceDdaNvsTreeThreadedAlgo();

  ncclResult_t allReduce() override;
};

} // namespace algorithms
} // namespace nccl
