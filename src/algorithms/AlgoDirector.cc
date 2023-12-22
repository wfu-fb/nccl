// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "AlgoDirector.h"

#include "AlgoUtils.h"
#include "DdaThreadedData.h"
#include "argcheck.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "nccl_cvars.h"

#include <cassert>

namespace nccl {
namespace algorithms {

/**
 * per communicator per rank Algorithm Manager that
 * - manages all the available algorithm instances for a given collective
 * - selects an optimal algorithm based on the input and environments
 */
AlgoDirector::AlgoDirector(ncclComm_t comm) : comm_(comm) {
  // register rank
  DdaThreadedData::get()->registerRank(comm->commHash, comm->rank);

  // enable peer access (support for NVS full-mesh topology only)
  for (int i = 0; i < comm->nRanks; ++i) {
    if (i == comm->rank) {
      continue;
    }
    cudaError_t e = cudaDeviceEnablePeerAccess(i, 0);
    if (e != cudaErrorPeerAccessAlreadyEnabled && e != cudaSuccess) {
      CUDACHECKIGNORE(e);
    }
  }

  this->allReduce = std::unique_ptr<AlgoManagerAllReduce>(new AlgoManagerAllReduce(comm));

  INFO(NCCL_INIT, "AlgoDirector initialized.");
}

AlgoDirector::~AlgoDirector() {
  // unregister rank
  DdaThreadedData::get()->unregisterRank(comm_->commHash, comm_->rank);
}

} // namespace algorithms
} // namespace nccl
