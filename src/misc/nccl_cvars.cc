// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Automatically generated by ./maint/extractcvars.py --- START
// DO NOT EDIT!!!

#include <iostream>
#include <unordered_set>
#include <string>
#include <vector>
#include "nccl_cvars.h"
#include "nccl_cvars_base.h"

int64_t NCCL_AGG_CHANNEL_SIZE;
int64_t NCCL_AGG_CHANNEL_SIZE_DEFAULT;
enum NCCL_ALLGATHER_ALGO NCCL_ALLGATHER_ALGO;
enum NCCL_ALLGATHER_ALGO NCCL_ALLGATHER_ALGO_DEFAULT;
uint64_t NCCL_ALLGATHER_DIRECT_CUTOFF;
uint64_t NCCL_ALLGATHER_DIRECT_CUTOFF_DEFAULT;
int64_t NCCL_ALLOC_P2P_NET_LL_BUFFERS;
int64_t NCCL_ALLOC_P2P_NET_LL_BUFFERS_DEFAULT;
enum NCCL_ALLREDUCE_ALGO NCCL_ALLREDUCE_ALGO;
enum NCCL_ALLREDUCE_ALGO NCCL_ALLREDUCE_ALGO_DEFAULT;
enum NCCL_ALLREDUCE_ALGO2 NCCL_ALLREDUCE_ALGO2;
enum NCCL_ALLREDUCE_ALGO2 NCCL_ALLREDUCE_ALGO2_DEFAULT;
int NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS;
int NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS_DEFAULT;
int NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE;
int NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE_DEFAULT;
int64_t NCCL_BUFFSIZE;
int64_t NCCL_BUFFSIZE_DEFAULT;
int64_t NCCL_CHECK_POINTERS;
int64_t NCCL_CHECK_POINTERS_DEFAULT;
int64_t NCCL_CHUNK_SIZE;
int64_t NCCL_CHUNK_SIZE_DEFAULT;
int64_t NCCL_COLLNET_NODE_THRESHOLD;
int64_t NCCL_COLLNET_NODE_THRESHOLD_DEFAULT;
int64_t NCCL_CONNECT_ROUND_SIZE;
int64_t NCCL_CONNECT_ROUND_SIZE_DEFAULT;
int64_t NCCL_CREATE_THREAD_CONTEXT;
int64_t NCCL_CREATE_THREAD_CONTEXT_DEFAULT;
int64_t NCCL_CROSS_NIC;
int64_t NCCL_CROSS_NIC_DEFAULT;
std::vector<enum NCCL_CTRAN_BACKENDS> NCCL_CTRAN_BACKENDS;
std::vector<enum NCCL_CTRAN_BACKENDS> NCCL_CTRAN_BACKENDS_DEFAULT;
int NCCL_CTRAN_IB_MAX_QPS;
int NCCL_CTRAN_IB_MAX_QPS_DEFAULT;
uint64_t NCCL_CTRAN_IB_QP_SCALING_THRESHOLD;
uint64_t NCCL_CTRAN_IB_QP_SCALING_THRESHOLD_DEFAULT;
bool NCCL_CTRAN_IB_TRAFFIC_PROFILNG;
bool NCCL_CTRAN_IB_TRAFFIC_PROFILNG_DEFAULT;
std::string NCCL_CTRAN_KINETO_PROFILE_DIR;
std::string NCCL_CTRAN_KINETO_PROFILE_DIR_DEFAULT;
enum NCCL_CTRAN_PROFILING NCCL_CTRAN_PROFILING;
enum NCCL_CTRAN_PROFILING NCCL_CTRAN_PROFILING_DEFAULT;
int NCCL_CTRAN_PROFILING_REPORT_COUNT;
int NCCL_CTRAN_PROFILING_REPORT_COUNT_DEFAULT;
enum NCCL_CTRAN_REGISTER NCCL_CTRAN_REGISTER;
enum NCCL_CTRAN_REGISTER NCCL_CTRAN_REGISTER_DEFAULT;
int NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT;
int NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT_DEFAULT;
std::string NCCL_CTRAN_TOPO_FILE;
std::string NCCL_CTRAN_TOPO_FILE_DEFAULT;
std::vector<std::string> NCCL_CTRAN_TOPO_FILE_KEYS;
std::vector<std::string> NCCL_CTRAN_TOPO_FILE_KEYS_DEFAULT;
int64_t NCCL_CUMEM_ENABLE;
int64_t NCCL_CUMEM_ENABLE_DEFAULT;
int NCCL_DDA2_ALLREDUCE_MAX_BLOCKS;
int NCCL_DDA2_ALLREDUCE_MAX_BLOCKS_DEFAULT;
uint64_t NCCL_DDA2_ALLREDUCE_SCATGAT_THRESHOLD;
uint64_t NCCL_DDA2_ALLREDUCE_SCATGAT_THRESHOLD_DEFAULT;
uint64_t NCCL_DDA2_ALLREDUCE_TREE_THRESHOLD;
uint64_t NCCL_DDA2_ALLREDUCE_TREE_THRESHOLD_DEFAULT;
uint64_t NCCL_DDA2_TMPBUFF_SIZE;
uint64_t NCCL_DDA2_TMPBUFF_SIZE_DEFAULT;
bool NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM;
bool NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM_DEFAULT;
int NCCL_DDA_ALLREDUCE_MAX_BLOCKS;
int NCCL_DDA_ALLREDUCE_MAX_BLOCKS_DEFAULT;
uint64_t NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE;
uint64_t NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE_DEFAULT;
uint64_t NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM;
uint64_t NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM_DEFAULT;
uint64_t NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS;
uint64_t NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS_DEFAULT;
bool NCCL_DDA_FORCE_P2P_ACCESS;
bool NCCL_DDA_FORCE_P2P_ACCESS_DEFAULT;
int NCCL_DDA_MAX_RANKS;
int NCCL_DDA_MAX_RANKS_DEFAULT;
int64_t NCCL_DMABUF_ENABLE;
int64_t NCCL_DMABUF_ENABLE_DEFAULT;
int64_t NCCL_GDRCOPY_ENABLE;
int64_t NCCL_GDRCOPY_ENABLE_DEFAULT;
int64_t NCCL_GDRCOPY_FIFO_ENABLE;
int64_t NCCL_GDRCOPY_FIFO_ENABLE_DEFAULT;
int64_t NCCL_GDRCOPY_FLUSH_ENABLE;
int64_t NCCL_GDRCOPY_FLUSH_ENABLE_DEFAULT;
int64_t NCCL_GDRCOPY_SYNC_ENABLE;
int64_t NCCL_GDRCOPY_SYNC_ENABLE_DEFAULT;
int64_t NCCL_GDR_FLUSH_DISABLE;
int64_t NCCL_GDR_FLUSH_DISABLE_DEFAULT;
int64_t NCCL_GRAPH_DUMP_FILE_RANK;
int64_t NCCL_GRAPH_DUMP_FILE_RANK_DEFAULT;
int64_t NCCL_GRAPH_MIXING_SUPPORT;
int64_t NCCL_GRAPH_MIXING_SUPPORT_DEFAULT;
int64_t NCCL_IB_ADAPTIVE_ROUTING;
int64_t NCCL_IB_ADAPTIVE_ROUTING_DEFAULT;
int64_t NCCL_IB_AR_THRESHOLD;
int64_t NCCL_IB_AR_THRESHOLD_DEFAULT;
int64_t NCCL_IB_DISABLE;
int64_t NCCL_IB_DISABLE_DEFAULT;
int64_t NCCL_IB_GID_INDEX;
int64_t NCCL_IB_GID_INDEX_DEFAULT;
std::string NCCL_IB_HCA_PREFIX;
std::string NCCL_IB_HCA_PREFIX_DEFAULT;
std::vector<std::string> NCCL_IB_HCA;
std::vector<std::string> NCCL_IB_HCA_DEFAULT;
int64_t NCCL_IB_MERGE_VFS;
int64_t NCCL_IB_MERGE_VFS_DEFAULT;
int64_t NCCL_IB_PCI_RELAXED_ORDERING;
int64_t NCCL_IB_PCI_RELAXED_ORDERING_DEFAULT;
int64_t NCCL_IB_PKEY;
int64_t NCCL_IB_PKEY_DEFAULT;
int64_t NCCL_IB_QPS_PER_CONNECTION;
int64_t NCCL_IB_QPS_PER_CONNECTION_DEFAULT;
int64_t NCCL_IB_RETRY_CNT;
int64_t NCCL_IB_RETRY_CNT_DEFAULT;
int64_t NCCL_IB_SL;
int64_t NCCL_IB_SL_DEFAULT;
int64_t NCCL_IB_SPLIT_DATA_ON_QPS;
int64_t NCCL_IB_SPLIT_DATA_ON_QPS_DEFAULT;
int64_t NCCL_IB_TC;
int64_t NCCL_IB_TC_DEFAULT;
int64_t NCCL_IB_TIMEOUT;
int64_t NCCL_IB_TIMEOUT_DEFAULT;
int64_t NCCL_IB_USE_INLINE;
int64_t NCCL_IB_USE_INLINE_DEFAULT;
int64_t NCCL_IGNORE_CPU_AFFINITY;
int64_t NCCL_IGNORE_CPU_AFFINITY_DEFAULT;
int64_t NCCL_IGNORE_DISABLED_P2P;
int64_t NCCL_IGNORE_DISABLED_P2P_DEFAULT;
int64_t NCCL_LL128_BUFFSIZE;
int64_t NCCL_LL128_BUFFSIZE_DEFAULT;
int64_t NCCL_LL128_NTHREADS;
int64_t NCCL_LL128_NTHREADS_DEFAULT;
int64_t NCCL_LL_BUFFSIZE;
int64_t NCCL_LL_BUFFSIZE_DEFAULT;
int64_t NCCL_LOCAL_REGISTER;
int64_t NCCL_LOCAL_REGISTER_DEFAULT;
int64_t NCCL_MAX_NCHANNELS;
int64_t NCCL_MAX_NCHANNELS_DEFAULT;
int64_t NCCL_MAX_NRINGS;
int64_t NCCL_MAX_NRINGS_DEFAULT;
int64_t NCCL_MAX_P2P_NCHANNELS;
int64_t NCCL_MAX_P2P_NCHANNELS_DEFAULT;
int64_t NCCL_MIN_NCHANNELS;
int64_t NCCL_MIN_NCHANNELS_DEFAULT;
int64_t NCCL_MIN_NRINGS;
int64_t NCCL_MIN_NRINGS_DEFAULT;
int64_t NCCL_MIN_P2P_NCHANNELS;
int64_t NCCL_MIN_P2P_NCHANNELS_DEFAULT;
int64_t NCCL_NCHANNELS_PER_NET_PEER;
int64_t NCCL_NCHANNELS_PER_NET_PEER_DEFAULT;
int64_t NCCL_NET_DISABLE_INTRA;
int64_t NCCL_NET_DISABLE_INTRA_DEFAULT;
int64_t NCCL_NET_FORCE_FLUSH;
int64_t NCCL_NET_FORCE_FLUSH_DEFAULT;
int64_t NCCL_NET_GDR_READ;
int64_t NCCL_NET_GDR_READ_DEFAULT;
int64_t NCCL_NET_OVERHEAD;
int64_t NCCL_NET_OVERHEAD_DEFAULT;
int64_t NCCL_NET_SHARED_BUFFERS;
int64_t NCCL_NET_SHARED_BUFFERS_DEFAULT;
int64_t NCCL_NET_SHARED_COMMS;
int64_t NCCL_NET_SHARED_COMMS_DEFAULT;
int64_t NCCL_NSOCKS_PERTHREAD;
int64_t NCCL_NSOCKS_PERTHREAD_DEFAULT;
int64_t NCCL_NTHREADS;
int64_t NCCL_NTHREADS_DEFAULT;
int64_t NCCL_NVB_DISABLE;
int64_t NCCL_NVB_DISABLE_DEFAULT;
int64_t NCCL_NVB_PRECONNECT;
int64_t NCCL_NVB_PRECONNECT_DEFAULT;
int64_t NCCL_P2P_NET_CHUNKSIZE;
int64_t NCCL_P2P_NET_CHUNKSIZE_DEFAULT;
int64_t NCCL_P2P_NVL_CHUNKSIZE;
int64_t NCCL_P2P_NVL_CHUNKSIZE_DEFAULT;
int64_t NCCL_P2P_PCI_CHUNKSIZE;
int64_t NCCL_P2P_PCI_CHUNKSIZE_DEFAULT;
int64_t NCCL_P2P_PXN_LEVEL;
int64_t NCCL_P2P_PXN_LEVEL_DEFAULT;
int64_t NCCL_PROGRESS_APPENDOP_FREQ;
int64_t NCCL_PROGRESS_APPENDOP_FREQ_DEFAULT;
int64_t NCCL_PROXY_APPEND_BATCH_SIZE;
int64_t NCCL_PROXY_APPEND_BATCH_SIZE_DEFAULT;
int64_t NCCL_PROXY_DUMP_SIGNAL;
int64_t NCCL_PROXY_DUMP_SIGNAL_DEFAULT;
int64_t NCCL_PXN_DISABLE;
int64_t NCCL_PXN_DISABLE_DEFAULT;
enum NCCL_SENDRECV_ALGO NCCL_SENDRECV_ALGO;
enum NCCL_SENDRECV_ALGO NCCL_SENDRECV_ALGO_DEFAULT;
int64_t NCCL_SET_STACK_SIZE;
int64_t NCCL_SET_STACK_SIZE_DEFAULT;
int64_t NCCL_SOCKET_NTHREADS;
int64_t NCCL_SOCKET_NTHREADS_DEFAULT;
int64_t NCCL_TOPO_DUMP_FILE_RANK;
int64_t NCCL_TOPO_DUMP_FILE_RANK_DEFAULT;
int64_t NCCL_WORK_FIFO_DEPTH;
int64_t NCCL_WORK_FIFO_DEPTH_DEFAULT;

void initEnvSet(std::unordered_set<std::string>& env) {
  env.insert("NCCL_AGG_CHANNEL_SIZE");
  env.insert("NCCL_ALLGATHER_ALGO");
  env.insert("NCCL_ALLGATHER_DIRECT_CUTOFF");
  env.insert("NCCL_ALLOC_P2P_NET_LL_BUFFERS");
  env.insert("NCCL_ALLREDUCE_ALGO");
  env.insert("NCCL_ALLREDUCE_ALGO2");
  env.insert("NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS");
  env.insert("NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE");
  env.insert("NCCL_BUFFSIZE");
  env.insert("NCCL_CHECK_POINTERS");
  env.insert("NCCL_CHUNK_SIZE");
  env.insert("NCCL_COLLNET_NODE_THRESHOLD");
  env.insert("NCCL_CONNECT_ROUND_SIZE");
  env.insert("NCCL_CREATE_THREAD_CONTEXT");
  env.insert("NCCL_CROSS_NIC");
  env.insert("NCCL_CTRAN_BACKENDS");
  env.insert("NCCL_CTRAN_IB_MAX_QPS");
  env.insert("NCCL_CTRAN_IB_QP_SCALING_THRESHOLD");
  env.insert("NCCL_CTRAN_IB_TRAFFIC_PROFILNG");
  env.insert("NCCL_CTRAN_KINETO_PROFILE_DIR");
  env.insert("NCCL_CTRAN_PROFILING");
  env.insert("NCCL_CTRAN_PROFILING_REPORT_COUNT");
  env.insert("NCCL_CTRAN_REGISTER");
  env.insert("NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT");
  env.insert("NCCL_CTRAN_TOPO_FILE");
  env.insert("NCCL_CTRAN_TOPO_FILE_KEYS");
  env.insert("NCCL_CUMEM_ENABLE");
  env.insert("NCCL_DDA2_ALLREDUCE_MAX_BLOCKS");
  env.insert("NCCL_DDA2_ALLREDUCE_SCATGAT_THRESHOLD");
  env.insert("NCCL_DDA2_ALLREDUCE_TREE_THRESHOLD");
  env.insert("NCCL_DDA2_TMPBUFF_SIZE");
  env.insert("NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM");
  env.insert("NCCL_DDA_ALLREDUCE_MAX_BLOCKS");
  env.insert("NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE");
  env.insert("NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM");
  env.insert("NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS");
  env.insert("NCCL_DDA_FORCE_P2P_ACCESS");
  env.insert("NCCL_DDA_MAX_RANKS");
  env.insert("NCCL_DMABUF_ENABLE");
  env.insert("NCCL_GDRCOPY_ENABLE");
  env.insert("NCCL_GDRCOPY_FIFO_ENABLE");
  env.insert("NCCL_GDRCOPY_FLUSH_ENABLE");
  env.insert("NCCL_GDRCOPY_SYNC_ENABLE");
  env.insert("NCCL_GDR_FLUSH_DISABLE");
  env.insert("NCCL_GRAPH_DUMP_FILE_RANK");
  env.insert("NCCL_GRAPH_MIXING_SUPPORT");
  env.insert("NCCL_IB_ADAPTIVE_ROUTING");
  env.insert("NCCL_IB_AR_THRESHOLD");
  env.insert("NCCL_IB_DISABLE");
  env.insert("NCCL_IB_GID_INDEX");
  env.insert("NCCL_IB_HCA");
  env.insert("NCCL_IB_MERGE_VFS");
  env.insert("NCCL_IB_PCI_RELAXED_ORDERING");
  env.insert("NCCL_IB_PKEY");
  env.insert("NCCL_IB_QPS_PER_CONNECTION");
  env.insert("NCCL_IB_RETRY_CNT");
  env.insert("NCCL_IB_SL");
  env.insert("NCCL_IB_SPLIT_DATA_ON_QPS");
  env.insert("NCCL_IB_TC");
  env.insert("NCCL_IB_TIMEOUT");
  env.insert("NCCL_IB_USE_INLINE");
  env.insert("NCCL_IGNORE_CPU_AFFINITY");
  env.insert("NCCL_IGNORE_DISABLED_P2P");
  env.insert("NCCL_LL128_BUFFSIZE");
  env.insert("NCCL_LL128_NTHREADS");
  env.insert("NCCL_LL_BUFFSIZE");
  env.insert("NCCL_LOCAL_REGISTER");
  env.insert("NCCL_MAX_NCHANNELS");
  env.insert("NCCL_MAX_NRINGS");
  env.insert("NCCL_MAX_P2P_NCHANNELS");
  env.insert("NCCL_MIN_NCHANNELS");
  env.insert("NCCL_MIN_NRINGS");
  env.insert("NCCL_MIN_P2P_NCHANNELS");
  env.insert("NCCL_NCHANNELS_PER_NET_PEER");
  env.insert("NCCL_NET_DISABLE_INTRA");
  env.insert("NCCL_NET_FORCE_FLUSH");
  env.insert("NCCL_NET_GDR_READ");
  env.insert("NCCL_NET_OVERHEAD");
  env.insert("NCCL_NET_SHARED_BUFFERS");
  env.insert("NCCL_NET_SHARED_COMMS");
  env.insert("NCCL_NSOCKS_PERTHREAD");
  env.insert("NCCL_NTHREADS");
  env.insert("NCCL_NVB_DISABLE");
  env.insert("NCCL_NVB_PRECONNECT");
  env.insert("NCCL_P2P_NET_CHUNKSIZE");
  env.insert("NCCL_P2P_NVL_CHUNKSIZE");
  env.insert("NCCL_P2P_PCI_CHUNKSIZE");
  env.insert("NCCL_P2P_PXN_LEVEL");
  env.insert("NCCL_PROGRESS_APPENDOP_FREQ");
  env.insert("NCCL_PROXY_APPEND_BATCH_SIZE");
  env.insert("NCCL_PROXY_DUMP_SIGNAL");
  env.insert("NCCL_PXN_DISABLE");
  env.insert("NCCL_SENDRECV_ALGO");
  env.insert("NCCL_SET_STACK_SIZE");
  env.insert("NCCL_SOCKET_NTHREADS");
  env.insert("NCCL_TOPO_DUMP_FILE_RANK");
  env.insert("NCCL_WORK_FIFO_DEPTH");
  env.insert("NCCL_ALGO");
  env.insert("NCCL_COLLNET_ENABLE");
  env.insert("NCCL_COLLTRACE_LOCAL_SUBDIR");
  env.insert("NCCL_COMM_ID");
  env.insert("NCCL_CUDA_PATH");
  env.insert("NCCL_CROSS_NIC");
  env.insert("NCCL_DEBUG");
  env.insert("NCCL_DEBUG_FILE");
  env.insert("NCCL_DEBUG_SUBSYS");
  env.insert("NCCL_GRAPH_DUMP_FILE");
  env.insert("NCCL_GRAPH_FILE");
  env.insert("NCCL_HOSTID");
  env.insert("NCCL_IB_DISABLE");
  env.insert("NCCL_IB_GID_INDEX");
  env.insert("NCCL_IB_TC");
  env.insert("NCCL_IB_TIMEOUT");
  env.insert("NCCL_IB_QPS_PER_CONNECTION");
  env.insert("NCCL_LAUNCH_MODE");
  env.insert("NCCL_NET");
  env.insert("NCCL_NET_PLUGIN");
  env.insert("NCCL_NET_SHARED_COMMS");
  env.insert("NCCL_NSOCKS_PERTHREAD");
  env.insert("NCCL_PROTO");
  env.insert("NCCL_PROXY_PROFILE");
  env.insert("NCCL_PXN_DISABLE");
  env.insert("NCCL_P2P_LEVEL");
  env.insert("NCCL_SHM_DISABLE");
  env.insert("NCCL_SOCKET_FAMILY");
  env.insert("NCCL_SOCKET_IFNAME");
  env.insert("NCCL_SOCKET_NTHREADS");
  env.insert("NCCL_THREAD_THRESHOLDS");
  env.insert("NCCL_TOPO_DUMP_FILE");
  env.insert("NCCL_TOPO_FILE");
  env.insert("NCCL_TUNER_PLUGIN");
}

void readCvarEnv() {
  NCCL_AGG_CHANNEL_SIZE = env2num<int64_t>("NCCL_AGG_CHANNEL_SIZE", "-2");
  NCCL_AGG_CHANNEL_SIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  if (getenv("NCCL_ALLGATHER_ALGO") == nullptr) {
    NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::orig;
  } else {
    std::string str(getenv("NCCL_ALLGATHER_ALGO"));
    if (str == std::string("orig")) {
      NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::orig;
    } else if (str == std::string("ctdirect")) {
      NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::ctdirect;
    } else if (str == std::string("ctring")) {
      NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::ctring;
    } else if (str == std::string("ctrd")) {
      NCCL_ALLGATHER_ALGO = NCCL_ALLGATHER_ALGO::ctrd;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_ALLGATHER_ALGO", str.c_str());
    }
  }
  NCCL_ALLGATHER_ALGO_DEFAULT = NCCL_ALLGATHER_ALGO::orig;

  NCCL_ALLGATHER_DIRECT_CUTOFF = env2num<uint64_t>("NCCL_ALLGATHER_DIRECT_CUTOFF", "524288");
  NCCL_ALLGATHER_DIRECT_CUTOFF_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "524288");

  NCCL_ALLOC_P2P_NET_LL_BUFFERS = env2num<int64_t>("NCCL_ALLOC_P2P_NET_LL_BUFFERS", "0");
  NCCL_ALLOC_P2P_NET_LL_BUFFERS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  if (getenv("NCCL_ALLREDUCE_ALGO") == nullptr) {
    NCCL_ALLREDUCE_ALGO = NCCL_ALLREDUCE_ALGO::orig;
  } else {
    std::string str(getenv("NCCL_ALLREDUCE_ALGO"));
    if (str == std::string("orig")) {
      NCCL_ALLREDUCE_ALGO = NCCL_ALLREDUCE_ALGO::orig;
    } else if (str == std::string("dda")) {
      NCCL_ALLREDUCE_ALGO = NCCL_ALLREDUCE_ALGO::dda;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_ALLREDUCE_ALGO", str.c_str());
    }
  }
  NCCL_ALLREDUCE_ALGO_DEFAULT = NCCL_ALLREDUCE_ALGO::orig;

  if (getenv("NCCL_ALLREDUCE_ALGO2") == nullptr) {
    NCCL_ALLREDUCE_ALGO2 = NCCL_ALLREDUCE_ALGO2::orig;
  } else {
    std::string str(getenv("NCCL_ALLREDUCE_ALGO2"));
    if (str == std::string("orig")) {
      NCCL_ALLREDUCE_ALGO2 = NCCL_ALLREDUCE_ALGO2::orig;
    } else if (str == std::string("dda")) {
      NCCL_ALLREDUCE_ALGO2 = NCCL_ALLREDUCE_ALGO2::dda;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_ALLREDUCE_ALGO2", str.c_str());
    }
  }
  NCCL_ALLREDUCE_ALGO2_DEFAULT = NCCL_ALLREDUCE_ALGO2::orig;

  NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS = env2num<int>("NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS", "-1");
  NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "-1");

  NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE = env2num<int>("NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE", "-1");
  NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "-1");

  NCCL_BUFFSIZE = env2num<int64_t>("NCCL_BUFFSIZE", "-2");
  NCCL_BUFFSIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_CHECK_POINTERS = env2num<int64_t>("NCCL_CHECK_POINTERS", "0");
  NCCL_CHECK_POINTERS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_CHUNK_SIZE = env2num<int64_t>("NCCL_CHUNK_SIZE", "0");
  NCCL_CHUNK_SIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_COLLNET_NODE_THRESHOLD = env2num<int64_t>("NCCL_COLLNET_NODE_THRESHOLD", "2");
  NCCL_COLLNET_NODE_THRESHOLD_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "2");

  NCCL_CONNECT_ROUND_SIZE = env2num<int64_t>("NCCL_CONNECT_ROUND_SIZE", "128");
  NCCL_CONNECT_ROUND_SIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "128");

  NCCL_CREATE_THREAD_CONTEXT = env2num<int64_t>("NCCL_CREATE_THREAD_CONTEXT", "0");
  NCCL_CREATE_THREAD_CONTEXT_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_CROSS_NIC = env2num<int64_t>("NCCL_CROSS_NIC", "2");
  NCCL_CROSS_NIC_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "2");

  {
    NCCL_CTRAN_BACKENDS.clear();
    auto tokens = env2strlist("NCCL_CTRAN_BACKENDS", "ib");
    for (auto token : tokens) {
      if (token == std::string("ib")) {
        NCCL_CTRAN_BACKENDS.emplace_back(NCCL_CTRAN_BACKENDS::ib);
      } else {
        CVAR_WARN_UNKNOWN_VALUE("NCCL_CTRAN_BACKENDS", token.c_str());
      }
    }
  }
  NCCL_CTRAN_BACKENDS_DEFAULT.clear();
  NCCL_CTRAN_BACKENDS_DEFAULT.emplace_back(NCCL_CTRAN_BACKENDS::ib);

  NCCL_CTRAN_IB_MAX_QPS = env2num<int>("NCCL_CTRAN_IB_MAX_QPS", "1");
  NCCL_CTRAN_IB_MAX_QPS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_CTRAN_IB_QP_SCALING_THRESHOLD = env2num<uint64_t>("NCCL_CTRAN_IB_QP_SCALING_THRESHOLD", "1048576");
  NCCL_CTRAN_IB_QP_SCALING_THRESHOLD_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "1048576");

  NCCL_CTRAN_IB_TRAFFIC_PROFILNG = env2bool("NCCL_CTRAN_IB_TRAFFIC_PROFILNG", "False");
  NCCL_CTRAN_IB_TRAFFIC_PROFILNG_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "False");

  NCCL_CTRAN_KINETO_PROFILE_DIR = env2str("NCCL_CTRAN_KINETO_PROFILE_DIR", "/tmp");
  NCCL_CTRAN_KINETO_PROFILE_DIR_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "/tmp");

  if (getenv("NCCL_CTRAN_PROFILING") == nullptr) {
    NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::none;
  } else {
    std::string str(getenv("NCCL_CTRAN_PROFILING"));
    if (str == std::string("none")) {
      NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::none;
    } else if (str == std::string("stdout")) {
      NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::stdout;
    } else if (str == std::string("info")) {
      NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::info;
    } else if (str == std::string("kineto")) {
      NCCL_CTRAN_PROFILING = NCCL_CTRAN_PROFILING::kineto;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_CTRAN_PROFILING", str.c_str());
    }
  }
  NCCL_CTRAN_PROFILING_DEFAULT = NCCL_CTRAN_PROFILING::none;

  NCCL_CTRAN_PROFILING_REPORT_COUNT = env2num<int>("NCCL_CTRAN_PROFILING_REPORT_COUNT", "100");
  NCCL_CTRAN_PROFILING_REPORT_COUNT_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "100");

  if (getenv("NCCL_CTRAN_REGISTER") == nullptr) {
    NCCL_CTRAN_REGISTER = NCCL_CTRAN_REGISTER::lazy;
  } else {
    std::string str(getenv("NCCL_CTRAN_REGISTER"));
    if (str == std::string("none")) {
      NCCL_CTRAN_REGISTER = NCCL_CTRAN_REGISTER::none;
    } else if (str == std::string("lazy")) {
      NCCL_CTRAN_REGISTER = NCCL_CTRAN_REGISTER::lazy;
    } else if (str == std::string("eager")) {
      NCCL_CTRAN_REGISTER = NCCL_CTRAN_REGISTER::eager;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_CTRAN_REGISTER", str.c_str());
    }
  }
  NCCL_CTRAN_REGISTER_DEFAULT = NCCL_CTRAN_REGISTER::lazy;

  NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT = env2num<int>("NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT", "-1");
  NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "-1");

  NCCL_CTRAN_TOPO_FILE = env2str("NCCL_CTRAN_TOPO_FILE", "");
  NCCL_CTRAN_TOPO_FILE_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "");

  NCCL_CTRAN_TOPO_FILE_KEYS.clear();
  NCCL_CTRAN_TOPO_FILE_KEYS = env2strlist("NCCL_CTRAN_TOPO_FILE_KEYS", "");
  NCCL_CTRAN_TOPO_FILE_KEYS_DEFAULT.clear();
  NCCL_CTRAN_TOPO_FILE_KEYS_DEFAULT = env2strlist("NCCL_ENV_DO_NOT_SET", "");

  NCCL_CUMEM_ENABLE = env2num<int64_t>("NCCL_CUMEM_ENABLE", "0");
  NCCL_CUMEM_ENABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_DDA2_ALLREDUCE_MAX_BLOCKS = env2num<int>("NCCL_DDA2_ALLREDUCE_MAX_BLOCKS", "24");
  NCCL_DDA2_ALLREDUCE_MAX_BLOCKS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "24");

  NCCL_DDA2_ALLREDUCE_SCATGAT_THRESHOLD = env2num<uint64_t>("NCCL_DDA2_ALLREDUCE_SCATGAT_THRESHOLD", "1048576");
  NCCL_DDA2_ALLREDUCE_SCATGAT_THRESHOLD_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "1048576");

  NCCL_DDA2_ALLREDUCE_TREE_THRESHOLD = env2num<uint64_t>("NCCL_DDA2_ALLREDUCE_TREE_THRESHOLD", "262144");
  NCCL_DDA2_ALLREDUCE_TREE_THRESHOLD_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "262144");

  NCCL_DDA2_TMPBUFF_SIZE = env2num<uint64_t>("NCCL_DDA2_TMPBUFF_SIZE", "33554432");
  NCCL_DDA2_TMPBUFF_SIZE_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "33554432");

  NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM = env2bool("NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM", "False");
  NCCL_DDA_ALLREDUCE_LARGE_MESSAGE_HCM_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "False");

  NCCL_DDA_ALLREDUCE_MAX_BLOCKS = env2num<int>("NCCL_DDA_ALLREDUCE_MAX_BLOCKS", "1");
  NCCL_DDA_ALLREDUCE_MAX_BLOCKS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE = env2num<uint64_t>("NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE", "33554432");
  NCCL_DDA_ALLREDUCE_TMPBUFF_SIZE_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "33554432");

  NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM = env2num<uint64_t>("NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM", "65536");
  NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_HCM_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "65536");

  NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS = env2num<uint64_t>("NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS", "262144");
  NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_NVS_DEFAULT = env2num<uint64_t>("NCCL_ENV_DO_NOT_SET", "262144");

  NCCL_DDA_FORCE_P2P_ACCESS = env2bool("NCCL_DDA_FORCE_P2P_ACCESS", "False");
  NCCL_DDA_FORCE_P2P_ACCESS_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "False");

  NCCL_DDA_MAX_RANKS = env2num<int>("NCCL_DDA_MAX_RANKS", "16");
  NCCL_DDA_MAX_RANKS_DEFAULT = env2num<int>("NCCL_ENV_DO_NOT_SET", "16");

  NCCL_DMABUF_ENABLE = env2num<int64_t>("NCCL_DMABUF_ENABLE", "1");
  NCCL_DMABUF_ENABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_GDRCOPY_ENABLE = env2num<int64_t>("NCCL_GDRCOPY_ENABLE", "0");
  NCCL_GDRCOPY_ENABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_GDRCOPY_FIFO_ENABLE = env2num<int64_t>("NCCL_GDRCOPY_FIFO_ENABLE", "-2");
  NCCL_GDRCOPY_FIFO_ENABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_GDRCOPY_FLUSH_ENABLE = env2num<int64_t>("NCCL_GDRCOPY_FLUSH_ENABLE", "0");
  NCCL_GDRCOPY_FLUSH_ENABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_GDRCOPY_SYNC_ENABLE = env2num<int64_t>("NCCL_GDRCOPY_SYNC_ENABLE", "1");
  NCCL_GDRCOPY_SYNC_ENABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_GDR_FLUSH_DISABLE = env2num<int64_t>("NCCL_GDR_FLUSH_DISABLE", "0");
  NCCL_GDR_FLUSH_DISABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_GRAPH_DUMP_FILE_RANK = env2num<int64_t>("NCCL_GRAPH_DUMP_FILE_RANK", "0");
  NCCL_GRAPH_DUMP_FILE_RANK_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_GRAPH_MIXING_SUPPORT = env2num<int64_t>("NCCL_GRAPH_MIXING_SUPPORT", "1");
  NCCL_GRAPH_MIXING_SUPPORT_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_IB_ADAPTIVE_ROUTING = env2num<int64_t>("NCCL_IB_ADAPTIVE_ROUTING", "-2");
  NCCL_IB_ADAPTIVE_ROUTING_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_IB_AR_THRESHOLD = env2num<int64_t>("NCCL_IB_AR_THRESHOLD", "8192");
  NCCL_IB_AR_THRESHOLD_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "8192");

  NCCL_IB_DISABLE = env2num<int64_t>("NCCL_IB_DISABLE", "0");
  NCCL_IB_DISABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_IB_GID_INDEX = env2num<int64_t>("NCCL_IB_GID_INDEX", "0");
  NCCL_IB_GID_INDEX_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  std::vector<std::string> NCCL_IB_HCA_allPrefixes{"^", "="};
  NCCL_IB_HCA.clear();
  std::tie(NCCL_IB_HCA_PREFIX, NCCL_IB_HCA) = env2prefixedStrlist("NCCL_IB_HCA", "", NCCL_IB_HCA_allPrefixes);
  NCCL_IB_HCA_DEFAULT.clear();
  std::tie(NCCL_IB_HCA_PREFIX_DEFAULT, NCCL_IB_HCA_DEFAULT) = env2prefixedStrlist("NCCL_ENV_DO_NOT_SET", "", NCCL_IB_HCA_allPrefixes);

  NCCL_IB_MERGE_VFS = env2num<int64_t>("NCCL_IB_MERGE_VFS", "1");
  NCCL_IB_MERGE_VFS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_IB_PCI_RELAXED_ORDERING = env2num<int64_t>("NCCL_IB_PCI_RELAXED_ORDERING", "2");
  NCCL_IB_PCI_RELAXED_ORDERING_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "2");

  NCCL_IB_PKEY = env2num<int64_t>("NCCL_IB_PKEY", "0");
  NCCL_IB_PKEY_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_IB_QPS_PER_CONNECTION = env2num<int64_t>("NCCL_IB_QPS_PER_CONNECTION", "1");
  NCCL_IB_QPS_PER_CONNECTION_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_IB_RETRY_CNT = env2num<int64_t>("NCCL_IB_RETRY_CNT", "7");
  NCCL_IB_RETRY_CNT_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "7");

  NCCL_IB_SL = env2num<int64_t>("NCCL_IB_SL", "0");
  NCCL_IB_SL_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_IB_SPLIT_DATA_ON_QPS = env2num<int64_t>("NCCL_IB_SPLIT_DATA_ON_QPS", "1");
  NCCL_IB_SPLIT_DATA_ON_QPS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_IB_TC = env2num<int64_t>("NCCL_IB_TC", "0");
  NCCL_IB_TC_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_IB_TIMEOUT = env2num<int64_t>("NCCL_IB_TIMEOUT", "18");
  NCCL_IB_TIMEOUT_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "18");

  NCCL_IB_USE_INLINE = env2num<int64_t>("NCCL_IB_USE_INLINE", "0");
  NCCL_IB_USE_INLINE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_IGNORE_CPU_AFFINITY = env2num<int64_t>("NCCL_IGNORE_CPU_AFFINITY", "0");
  NCCL_IGNORE_CPU_AFFINITY_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_IGNORE_DISABLED_P2P = env2num<int64_t>("NCCL_IGNORE_DISABLED_P2P", "0");
  NCCL_IGNORE_DISABLED_P2P_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_LL128_BUFFSIZE = env2num<int64_t>("NCCL_LL128_BUFFSIZE", "-2");
  NCCL_LL128_BUFFSIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_LL128_NTHREADS = env2num<int64_t>("NCCL_LL128_NTHREADS", "-2");
  NCCL_LL128_NTHREADS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_LL_BUFFSIZE = env2num<int64_t>("NCCL_LL_BUFFSIZE", "-2");
  NCCL_LL_BUFFSIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_LOCAL_REGISTER = env2num<int64_t>("NCCL_LOCAL_REGISTER", "1");
  NCCL_LOCAL_REGISTER_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_MAX_NCHANNELS = env2num<int64_t>("NCCL_MAX_NCHANNELS", "-2");
  NCCL_MAX_NCHANNELS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_MAX_NRINGS = env2num<int64_t>("NCCL_MAX_NRINGS", "-2");
  NCCL_MAX_NRINGS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_MAX_P2P_NCHANNELS = env2num<int64_t>("NCCL_MAX_P2P_NCHANNELS", "32");
  NCCL_MAX_P2P_NCHANNELS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "32");

  NCCL_MIN_NCHANNELS = env2num<int64_t>("NCCL_MIN_NCHANNELS", "-2");
  NCCL_MIN_NCHANNELS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_MIN_NRINGS = env2num<int64_t>("NCCL_MIN_NRINGS", "-2");
  NCCL_MIN_NRINGS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_MIN_P2P_NCHANNELS = env2num<int64_t>("NCCL_MIN_P2P_NCHANNELS", "1");
  NCCL_MIN_P2P_NCHANNELS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_NCHANNELS_PER_NET_PEER = env2num<int64_t>("NCCL_NCHANNELS_PER_NET_PEER", "2");
  NCCL_NCHANNELS_PER_NET_PEER_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "2");

  NCCL_NET_DISABLE_INTRA = env2num<int64_t>("NCCL_NET_DISABLE_INTRA", "0");
  NCCL_NET_DISABLE_INTRA_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_NET_FORCE_FLUSH = env2num<int64_t>("NCCL_NET_FORCE_FLUSH", "1");
  NCCL_NET_FORCE_FLUSH_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_NET_GDR_READ = env2num<int64_t>("NCCL_NET_GDR_READ", "-2");
  NCCL_NET_GDR_READ_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_NET_OVERHEAD = env2num<int64_t>("NCCL_NET_OVERHEAD", "-2");
  NCCL_NET_OVERHEAD_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_NET_SHARED_BUFFERS = env2num<int64_t>("NCCL_NET_SHARED_BUFFERS", "-2");
  NCCL_NET_SHARED_BUFFERS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_NET_SHARED_COMMS = env2num<int64_t>("NCCL_NET_SHARED_COMMS", "1");
  NCCL_NET_SHARED_COMMS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_NSOCKS_PERTHREAD = env2num<int64_t>("NCCL_NSOCKS_PERTHREAD", "-2");
  NCCL_NSOCKS_PERTHREAD_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_NTHREADS = env2num<int64_t>("NCCL_NTHREADS", "-2");
  NCCL_NTHREADS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_NVB_DISABLE = env2num<int64_t>("NCCL_NVB_DISABLE", "0");
  NCCL_NVB_DISABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_NVB_PRECONNECT = env2num<int64_t>("NCCL_NVB_PRECONNECT", "1");
  NCCL_NVB_PRECONNECT_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "1");

  NCCL_P2P_NET_CHUNKSIZE = env2num<int64_t>("NCCL_P2P_NET_CHUNKSIZE", "131072");
  NCCL_P2P_NET_CHUNKSIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "131072");

  NCCL_P2P_NVL_CHUNKSIZE = env2num<int64_t>("NCCL_P2P_NVL_CHUNKSIZE", "524288");
  NCCL_P2P_NVL_CHUNKSIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "524288");

  NCCL_P2P_PCI_CHUNKSIZE = env2num<int64_t>("NCCL_P2P_PCI_CHUNKSIZE", "131072");
  NCCL_P2P_PCI_CHUNKSIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "131072");

  NCCL_P2P_PXN_LEVEL = env2num<int64_t>("NCCL_P2P_PXN_LEVEL", "2");
  NCCL_P2P_PXN_LEVEL_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "2");

  NCCL_PROGRESS_APPENDOP_FREQ = env2num<int64_t>("NCCL_PROGRESS_APPENDOP_FREQ", "8");
  NCCL_PROGRESS_APPENDOP_FREQ_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "8");

  NCCL_PROXY_APPEND_BATCH_SIZE = env2num<int64_t>("NCCL_PROXY_APPEND_BATCH_SIZE", "16");
  NCCL_PROXY_APPEND_BATCH_SIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "16");

  NCCL_PROXY_DUMP_SIGNAL = env2num<int64_t>("NCCL_PROXY_DUMP_SIGNAL", "-1");
  NCCL_PROXY_DUMP_SIGNAL_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-1");

  NCCL_PXN_DISABLE = env2num<int64_t>("NCCL_PXN_DISABLE", "0");
  NCCL_PXN_DISABLE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  if (getenv("NCCL_SENDRECV_ALGO") == nullptr) {
    NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::orig;
  } else {
    std::string str(getenv("NCCL_SENDRECV_ALGO"));
    if (str == std::string("orig")) {
      NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::orig;
    } else if (str == std::string("ctran")) {
      NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::ctran;
    } else {
      CVAR_WARN_UNKNOWN_VALUE("NCCL_SENDRECV_ALGO", str.c_str());
    }
  }
  NCCL_SENDRECV_ALGO_DEFAULT = NCCL_SENDRECV_ALGO::orig;

  NCCL_SET_STACK_SIZE = env2num<int64_t>("NCCL_SET_STACK_SIZE", "0");
  NCCL_SET_STACK_SIZE_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_SOCKET_NTHREADS = env2num<int64_t>("NCCL_SOCKET_NTHREADS", "-2");
  NCCL_SOCKET_NTHREADS_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "-2");

  NCCL_TOPO_DUMP_FILE_RANK = env2num<int64_t>("NCCL_TOPO_DUMP_FILE_RANK", "0");
  NCCL_TOPO_DUMP_FILE_RANK_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "0");

  NCCL_WORK_FIFO_DEPTH = env2num<int64_t>("NCCL_WORK_FIFO_DEPTH", "65536");
  NCCL_WORK_FIFO_DEPTH_DEFAULT = env2num<int64_t>("NCCL_ENV_DO_NOT_SET", "65536");

}

// Automatically generated by ./maint/extractcvars.py --- END
