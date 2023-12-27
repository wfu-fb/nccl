// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <memory>

#include "AlgoManagerBase.h"
#include "AlgoAllReduceDdaNvsFlatThreaded.h"
#include "AlgoAllReduceDdaNvsTreeThreaded.h"
#include "AlgoAllReduceDdaNvsFlatIpc.h"
#include "AlgoAllReduceDdaNvsTreeIpc.h"
#include "AlgoAllReduceDdaNvsScatGatIpc.h"

namespace nccl {
namespace algorithms {

/**
 * per communicator per rank Algorithm Manager that
 * - manages all the available algorithm instances for a given collective
 * - selects an optimal algorithm based on the input and environments
 */
class AlgoManagerAllReduce : AlgoManagerBase {
 public:
  AlgoManagerAllReduce(ncclComm_t comm)
    : AlgoManagerBase(comm) {}
  ~AlgoManagerAllReduce() {}

  // get an optimal customized algorithm instance
  // return nullptr if no suitable algorithm is found (fallback to NV
  // implementation)
  std::unique_ptr<AlgoAllReduce> getAlgoAllReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm* comm,
      cudaStream_t stream);

  std::unique_ptr<AlgoAllReduceDdaNvsFlatThreaded>
  getAlgoAllReduceDdaNvsFlatThreaded(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm* comm,
      cudaStream_t stream);

  std::unique_ptr<AlgoAllReduceDdaNvsTreeThreaded>
  getAlgoAllReduceDdaNvsTreeThreaded(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm* comm,
      cudaStream_t stream);

  std::unique_ptr<AlgoAllReduceDdaNvsFlatIpc>
  getAlgoAllReduceDdaNvsFlatIpc(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm* comm,
      cudaStream_t stream);

  std::unique_ptr<AlgoAllReduceDdaNvsTreeIpc>
  getAlgoAllReduceDdaNvsTreeIpc(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm* comm,
      cudaStream_t stream);

  std::unique_ptr<AlgoAllReduceDdaNvsScatGatIpc>
  getAlgoAllReduceDdaNvsScatGatIpc(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm* comm,
      cudaStream_t stream);

  // check DDA threaded requirements
  static bool canRunDdaAllReduceThreaded(
    ncclComm* comm,
    ncclRedOp_t op,
    const void* sendbuff,
    void* recvbuff,
    size_t totalBytes,
    size_t numDdaThreads,
    size_t treeThresholdBytes);

  // check DDA IPC requirements
  static bool canRunDdaAllReduceIpc(
    ncclComm* comm,
    ncclRedOp_t op,
    const void* sendbuff,
    void* recvbuff,
    size_t totalBytes,
    size_t treeThresholdBytes,
    size_t tmpbuffSize);
};

} // namespace algorithms
} // namespace nccl
