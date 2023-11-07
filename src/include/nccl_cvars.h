// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Automatically generated
//   by ./maint/extractcvars.py
// DO NOT EDIT!!!

#ifndef NCCL_CVARS_H_INCLUDED
#define NCCL_CVARS_H_INCLUDED

#include <string>
#include <set>

extern bool NCCL_CVAR_DDA_ALLREDUCE_LARGE_MESSAGE_HCM;

extern int NCCL_CVAR_DDA_ALLREDUCE_TMPBUFF_SIZE;

extern int NCCL_CVAR_DDA_MAX_RANKS;

enum class NCCL_CVAR_ALLREDUCE_ALGO {
  orig,
  dda,
};
extern enum NCCL_CVAR_ALLREDUCE_ALGO NCCL_CVAR_ALLREDUCE_ALGO;

extern int NCCL_CVAR_DDA_ALLREDUCE_MAX_BLOCKS;

extern int NCCL_CVAR_DDA_ALLREDUCE_TREE_THRESHOLD_NVS;

extern int NCCL_CVAR_DDA_ALLREDUCE_TREE_THRESHOLD_HCM;

extern bool NCCL_CVAR_DDA_FORCE_P2P_ACCESS;

enum class NCCL_CVAR_SENDRECV_ALGO {
  orig,
  ctran,
};
extern enum NCCL_CVAR_SENDRECV_ALGO NCCL_CVAR_SENDRECV_ALGO;

extern int NCCL_CVAR_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS;

extern int NCCL_CVAR_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE;

extern int NCCL_CVAR_ALLGATHER_DIRECT_CUTOFF;

enum class NCCL_CVAR_ALLGATHER_ALGO {
  orig,
  ctdirect,
  ctring,
  ctrd,
};
extern enum NCCL_CVAR_ALLGATHER_ALGO NCCL_CVAR_ALLGATHER_ALGO;

extern bool NCCL_CVAR_CTRAN_ENABLE_LOCAL_IB;

extern std::set<std::string> NCCL_IB_HCA;

extern int NCCL_CVAR_CTRAN_IB_MAX_QPS;

extern int NCCL_CVAR_CTRAN_IB_QP_SCALING_THRESHOLD;

enum class NCCL_CVAR_CTRAN_PROFILING {
  none,
  stdout,
  kineto,
};
extern enum NCCL_CVAR_CTRAN_PROFILING NCCL_CVAR_CTRAN_PROFILING;

extern std::string NCCL_CVAR_CTRAN_KINETO_PROFILE_DIR;

enum class NCCL_CVAR_CTRAN_REGISTER {
  none,
  lazy,
  eager,
};
extern enum NCCL_CVAR_CTRAN_REGISTER NCCL_CVAR_CTRAN_REGISTER;

enum class NCCL_CVAR_CTRAN_BACKENDS {
  ib,
  nvl,
};
extern std::set<enum NCCL_CVAR_CTRAN_BACKENDS> NCCL_CVAR_CTRAN_BACKENDS;

extern int NCCL_CVAR_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT;

void ncclCvarInit();

#endif  /* NCCL_CVARS_H_INCLUDED */
