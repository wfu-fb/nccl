// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Automatically generated
//   by ./maint/extractcvars.py
// DO NOT EDIT!!!

#ifndef NCCL_CVARS_H_INCLUDED
#define NCCL_CVARS_H_INCLUDED

#include <string>
#include <set>

extern bool NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM;

extern int NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE;

extern int NCCL_DDA_MAX_RANKS;

enum class NCCL_ALLREDUCE_ALGO {
  orig,
  dda,
};
extern enum NCCL_ALLREDUCE_ALGO NCCL_ALLREDUCE_ALGO;

extern int NCCL_ALLGATHER_DIRECT_CUTOFF;

extern int NCCL_DDA_ALLREDUCE_MAX_BLOCKS;

extern int NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS;

extern int NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM;

extern int NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS;

extern int NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE;

extern bool NCCL_DDA_FORCE_P2P_ACCESS;

void ncclCvarInit();

#endif  /* NCCL_CVARS_H_INCLUDED */
