// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <memory>

#include "AllReduceDdaNvsFlatThreadedAlgo.h"
#include "AllReduceDdaNvsTreeThreadedAlgo.h"

namespace nccl {
namespace algorithms {

/**
 * per communicator per rank Algorithm Manager that
 * - manages all the available algorithm instances for a given collective
 * - selects an optimal algorithm based on the input and environments
 */
class AlgoManager {
 public:
  AlgoManager(ncclComm_t comm);

  std::unique_ptr<AllReduceAlgo> getAllReduceAlgo(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm* comm,
      cudaStream_t stream);

 private:
  ncclComm_t comm_{nullptr};
};

} // namespace algorithms
} // namespace nccl
