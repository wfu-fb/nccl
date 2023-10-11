// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include "comm.h"
#include "nccl.h"

namespace nccl {
namespace algorithms {

ncclResult_t algoInit(ncclComm_t comm);

} // namespace algorithms
} // namespace nccl
