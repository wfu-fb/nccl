// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "AllReduceDdaNvsTreeThreadedAlgo.h"

#include "debug.h"

namespace nccl {
namespace algorithms {

AllReduceDdaNvsTreeThreadedAlgo::AllReduceDdaNvsTreeThreadedAlgo() {}

AllReduceDdaNvsTreeThreadedAlgo::~AllReduceDdaNvsTreeThreadedAlgo() {}

ncclResult_t AllReduceDdaNvsTreeThreadedAlgo::allReduce() {
  INFO(NCCL_COLL, "AllReduceDdaNvsTreeThreadedAlgo::allReduce");
  return ncclSuccess;
}

} // namespace algorithms
} // namespace nccl
