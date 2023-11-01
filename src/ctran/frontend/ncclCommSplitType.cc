// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "nccl.h"
#include "argcheck.h"
#include "comm.h"

NCCL_API(ncclResult_t, ncclCommSplitType, ncclComm_t comm, int type, int key, ncclComm_t *newcomm, ncclConfig_t *config);
ncclResult_t ncclCommSplitType(ncclComm_t comm, int type, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  ncclResult_t res = ncclSuccess;
  int color;

  if (type == NCCL_SPLIT_TYPE_UNDEFINED) {
    color = NCCL_SPLIT_NOCOLOR;
  } else if (type == NCCL_SPLIT_TYPE_NODE) {
    color = comm->rankToNode[comm->localRank];
  } else {
    if (newcomm) *newcomm = NULL;
    return ncclInvalidArgument;
  }

  return ncclCommSplit(comm, color, key, newcomm, config);
}
