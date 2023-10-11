// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include "AllReduceAlgo.h"

namespace nccl {
namespace algorithms {

class AllReduceDdaNvsFlatThreadedAlgo : public AllReduceAlgo {
 public:
  AllReduceDdaNvsFlatThreadedAlgo();
  ~AllReduceDdaNvsFlatThreadedAlgo();

  ncclResult_t allReduce() override;
};

} // namespace algorithms
} // namespace nccl
