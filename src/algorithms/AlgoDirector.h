// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <memory>

#include "AlgoManagerAllReduce.h"

namespace nccl {
namespace algorithms {

/**
 * per communicator per rank Algorithm Manager that
 * - manages all the available algorithm instances for a given collective
 * - selects an optimal algorithm based on the input and environments
 */
class AlgoDirector {
 public:
  AlgoDirector(ncclComm_t comm);
  ~AlgoDirector();

  std::unique_ptr<AlgoManagerAllReduce> allReduce;

 private:
  ncclComm_t comm_;
};

} // namespace algorithms
} // namespace nccl
