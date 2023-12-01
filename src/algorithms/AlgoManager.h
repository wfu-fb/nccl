// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <memory>

#include "AllReduceDdaNvsFlatThreadedAlgo.h"
#include "AllReduceDdaNvsTreeThreadedAlgo.h"
#include "DdaMemHandler.h"
#include "collectives.h"

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

  ~AlgoManager();

  // get an optimal customized algorithm instance
  // return nullptr if no suitable algorithm is found (fallback to NV
  // implementation)
  std::unique_ptr<AllReduceAlgo> getAllReduceAlgo(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm* comm,
      cudaStream_t stream);

  std::unique_ptr<AllReduceDdaNvsFlatThreadedAlgo>
  getAllReduceDdaNvsFlatThreadedAlgo(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm* comm,
      cudaStream_t stream);

  std::unique_ptr<AllReduceDdaNvsTreeThreadedAlgo>
  getAllReduceDdaNvsTreeThreadedAlgo(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm* comm,
      cudaStream_t stream);

 private:
  ncclComm_t comm_{nullptr};
  cudaDeviceProp devProp_;
  DdaMemHandler memHandler_;

  // device buffers
  uintptr_t* barrierMbox_d_{nullptr};
  void* tmpbuff_d_{nullptr};
  DdaDeviceState* devStates_d_{nullptr};

  uintptr_t barrierFlag_{0};
};

} // namespace algorithms
} // namespace nccl
