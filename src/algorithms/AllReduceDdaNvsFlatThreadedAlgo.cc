// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "AllReduceDdaNvsFlatThreadedAlgo.h"

#include "debug.h"

namespace nccl {
namespace algorithms {

AllReduceDdaNvsFlatThreadedAlgo::AllReduceDdaNvsFlatThreadedAlgo() {}

AllReduceDdaNvsFlatThreadedAlgo::~AllReduceDdaNvsFlatThreadedAlgo() {}

ncclResult_t AllReduceDdaNvsFlatThreadedAlgo::allReduce() {
  INFO(NCCL_COLL, "AllReduceDdaNvsFlatThreadedAlgo::allReduce");
  return ncclSuccess;
}

} // namespace algorithms
} // namespace nccl
