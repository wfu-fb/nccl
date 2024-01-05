// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Automatically generated by ./maint/extractcvars.py --- START
// DO NOT EDIT!!!

#ifndef NCCL_CVARS_H_INCLUDED
#define NCCL_CVARS_H_INCLUDED

#include <string>
#include <vector>

extern std::string CUDA_LAUNCH_BLOCKING;
extern std::string CUDA_LAUNCH_BLOCKING_DEFAULT;

extern int64_t NCCL_AGG_CHANNEL_SIZE;
extern int64_t NCCL_AGG_CHANNEL_SIZE_DEFAULT;

extern std::string NCCL_ALGO;
extern std::string NCCL_ALGO_DEFAULT;

enum class NCCL_ALLGATHER_ALGO {
  orig,
  ctdirect,
  ctring,
  ctrd,
};
extern enum NCCL_ALLGATHER_ALGO NCCL_ALLGATHER_ALGO;
extern enum NCCL_ALLGATHER_ALGO NCCL_ALLGATHER_ALGO_DEFAULT;

extern uint64_t NCCL_ALLGATHER_DIRECT_CUTOFF;
extern uint64_t NCCL_ALLGATHER_DIRECT_CUTOFF_DEFAULT;

extern int64_t NCCL_ALLOC_P2P_NET_LL_BUFFERS;
extern int64_t NCCL_ALLOC_P2P_NET_LL_BUFFERS_DEFAULT;

enum class NCCL_ALLREDUCE_ALGO {
  orig,
  dda,
};
extern enum NCCL_ALLREDUCE_ALGO NCCL_ALLREDUCE_ALGO;
extern enum NCCL_ALLREDUCE_ALGO NCCL_ALLREDUCE_ALGO_DEFAULT;

extern int NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS;
extern int NCCL_ALLREDUCE_SPARSE_BLOCK_NUM_THREAD_BLOCKS_DEFAULT;

extern int NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE;
extern int NCCL_ALLREDUCE_SPARSE_BLOCK_THREAD_BLOCK_SIZE_DEFAULT;

extern int64_t NCCL_BUFFSIZE;
extern int64_t NCCL_BUFFSIZE_DEFAULT;

extern int64_t NCCL_CGA_CLUSTER_SIZE;
extern int64_t NCCL_CGA_CLUSTER_SIZE_DEFAULT;

extern int64_t NCCL_CHECK_POINTERS;
extern int64_t NCCL_CHECK_POINTERS_DEFAULT;

extern int64_t NCCL_CHUNK_SIZE;
extern int64_t NCCL_CHUNK_SIZE_DEFAULT;

extern std::string NCCL_COLLNET_ENABLE;
extern std::string NCCL_COLLNET_ENABLE_DEFAULT;

extern int64_t NCCL_COLLNET_NODE_THRESHOLD;
extern int64_t NCCL_COLLNET_NODE_THRESHOLD_DEFAULT;

extern std::string NCCL_COLLTRACE_LOCAL_SUBDIR;
extern std::string NCCL_COLLTRACE_LOCAL_SUBDIR_DEFAULT;

extern int64_t NCCL_COMM_BLOCKING;
extern int64_t NCCL_COMM_BLOCKING_DEFAULT;

extern std::string NCCL_COMM_ID;
extern std::string NCCL_COMM_ID_DEFAULT;

extern int64_t NCCL_COMM_SPLIT_SHARE_RESOURCES;
extern int64_t NCCL_COMM_SPLIT_SHARE_RESOURCES_DEFAULT;

extern int64_t NCCL_CONNECT_ROUND_SIZE;
extern int64_t NCCL_CONNECT_ROUND_SIZE_DEFAULT;

extern int64_t NCCL_CREATE_THREAD_CONTEXT;
extern int64_t NCCL_CREATE_THREAD_CONTEXT_DEFAULT;

extern int64_t NCCL_CROSS_NIC;
extern int64_t NCCL_CROSS_NIC_DEFAULT;

enum class NCCL_CTRAN_BACKENDS {
  ib,
};
extern std::vector<enum NCCL_CTRAN_BACKENDS> NCCL_CTRAN_BACKENDS;
extern std::vector<enum NCCL_CTRAN_BACKENDS> NCCL_CTRAN_BACKENDS_DEFAULT;

extern int NCCL_CTRAN_IB_MAX_QPS;
extern int NCCL_CTRAN_IB_MAX_QPS_DEFAULT;

extern uint64_t NCCL_CTRAN_IB_QP_SCALING_THRESHOLD;
extern uint64_t NCCL_CTRAN_IB_QP_SCALING_THRESHOLD_DEFAULT;

extern bool NCCL_CTRAN_IB_TRAFFIC_PROFILNG;
extern bool NCCL_CTRAN_IB_TRAFFIC_PROFILNG_DEFAULT;

extern std::string NCCL_CTRAN_KINETO_PROFILE_DIR;
extern std::string NCCL_CTRAN_KINETO_PROFILE_DIR_DEFAULT;

enum class NCCL_CTRAN_PROFILING {
  none,
  stdout,
  info,
  kineto,
};
extern enum NCCL_CTRAN_PROFILING NCCL_CTRAN_PROFILING;
extern enum NCCL_CTRAN_PROFILING NCCL_CTRAN_PROFILING_DEFAULT;

extern int NCCL_CTRAN_PROFILING_REPORT_COUNT;
extern int NCCL_CTRAN_PROFILING_REPORT_COUNT_DEFAULT;

enum class NCCL_CTRAN_REGISTER {
  none,
  lazy,
  eager,
};
extern enum NCCL_CTRAN_REGISTER NCCL_CTRAN_REGISTER;
extern enum NCCL_CTRAN_REGISTER NCCL_CTRAN_REGISTER_DEFAULT;

extern int NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT;
extern int NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT_DEFAULT;

extern int NCCL_CTRAN_RING_MAX_OUTSTANDING;
extern int NCCL_CTRAN_RING_MAX_OUTSTANDING_DEFAULT;

extern uint64_t NCCL_CTRAN_RING_STEP;
extern uint64_t NCCL_CTRAN_RING_STEP_DEFAULT;

extern std::string NCCL_CTRAN_TOPO_FILE;
extern std::string NCCL_CTRAN_TOPO_FILE_DEFAULT;

extern std::vector<std::string> NCCL_CTRAN_TOPO_FILE_KEYS;
extern std::vector<std::string> NCCL_CTRAN_TOPO_FILE_KEYS_DEFAULT;

extern std::string NCCL_CUDA_PATH;
extern std::string NCCL_CUDA_PATH_DEFAULT;

extern int64_t NCCL_CUMEM_ENABLE;
extern int64_t NCCL_CUMEM_ENABLE_DEFAULT;

extern int NCCL_DDA_ALLREDUCE_MAX_BLOCKS;
extern int NCCL_DDA_ALLREDUCE_MAX_BLOCKS_DEFAULT;

extern uint64_t NCCL_DDA_ALLREDUCE_SCATGAT_THRESHOLD;
extern uint64_t NCCL_DDA_ALLREDUCE_SCATGAT_THRESHOLD_DEFAULT;

extern uint64_t NCCL_DDA_ALLREDUCE_TREE_THRESHOLD;
extern uint64_t NCCL_DDA_ALLREDUCE_TREE_THRESHOLD_DEFAULT;

extern uint64_t NCCL_DDA_TMPBUFF_SIZE;
extern uint64_t NCCL_DDA_TMPBUFF_SIZE_DEFAULT;

extern std::string NCCL_DEBUG;
extern std::string NCCL_DEBUG_DEFAULT;

extern std::string NCCL_DEBUG_FILE;
extern std::string NCCL_DEBUG_FILE_DEFAULT;

extern std::string NCCL_DEBUG_SUBSYS;
extern std::string NCCL_DEBUG_SUBSYS_DEFAULT;

extern int64_t NCCL_DMABUF_ENABLE;
extern int64_t NCCL_DMABUF_ENABLE_DEFAULT;

extern int64_t NCCL_GDRCOPY_ENABLE;
extern int64_t NCCL_GDRCOPY_ENABLE_DEFAULT;

extern int64_t NCCL_GDRCOPY_FIFO_ENABLE;
extern int64_t NCCL_GDRCOPY_FIFO_ENABLE_DEFAULT;

extern int64_t NCCL_GDRCOPY_FLUSH_ENABLE;
extern int64_t NCCL_GDRCOPY_FLUSH_ENABLE_DEFAULT;

extern int64_t NCCL_GDRCOPY_SYNC_ENABLE;
extern int64_t NCCL_GDRCOPY_SYNC_ENABLE_DEFAULT;

extern int64_t NCCL_GDR_FLUSH_DISABLE;
extern int64_t NCCL_GDR_FLUSH_DISABLE_DEFAULT;

extern std::string NCCL_GRAPH_DUMP_FILE;
extern std::string NCCL_GRAPH_DUMP_FILE_DEFAULT;

extern int64_t NCCL_GRAPH_DUMP_FILE_RANK;
extern int64_t NCCL_GRAPH_DUMP_FILE_RANK_DEFAULT;

extern std::string NCCL_GRAPH_FILE;
extern std::string NCCL_GRAPH_FILE_DEFAULT;

extern int64_t NCCL_GRAPH_MIXING_SUPPORT;
extern int64_t NCCL_GRAPH_MIXING_SUPPORT_DEFAULT;

extern int64_t NCCL_GRAPH_REGISTER;
extern int64_t NCCL_GRAPH_REGISTER_DEFAULT;

extern std::string NCCL_HOSTID;
extern std::string NCCL_HOSTID_DEFAULT;

extern int64_t NCCL_IB_ADAPTIVE_ROUTING;
extern int64_t NCCL_IB_ADAPTIVE_ROUTING_DEFAULT;

extern int64_t NCCL_IB_AR_THRESHOLD;
extern int64_t NCCL_IB_AR_THRESHOLD_DEFAULT;

extern int64_t NCCL_IB_DISABLE;
extern int64_t NCCL_IB_DISABLE_DEFAULT;

extern int64_t NCCL_IB_GID_INDEX;
extern int64_t NCCL_IB_GID_INDEX_DEFAULT;

extern std::string NCCL_IB_HCA_PREFIX;
extern std::string NCCL_IB_HCA_PREFIX_DEFAULT;
extern std::vector<std::string> NCCL_IB_HCA;
extern std::vector<std::string> NCCL_IB_HCA_DEFAULT;

extern int64_t NCCL_IB_MERGE_VFS;
extern int64_t NCCL_IB_MERGE_VFS_DEFAULT;

extern int64_t NCCL_IB_PCI_RELAXED_ORDERING;
extern int64_t NCCL_IB_PCI_RELAXED_ORDERING_DEFAULT;

extern int64_t NCCL_IB_PKEY;
extern int64_t NCCL_IB_PKEY_DEFAULT;

extern int64_t NCCL_IB_QPS_PER_CONNECTION;
extern int64_t NCCL_IB_QPS_PER_CONNECTION_DEFAULT;

extern int64_t NCCL_IB_RETRY_CNT;
extern int64_t NCCL_IB_RETRY_CNT_DEFAULT;

extern int64_t NCCL_IB_SL;
extern int64_t NCCL_IB_SL_DEFAULT;

extern int64_t NCCL_IB_SPLIT_DATA_ON_QPS;
extern int64_t NCCL_IB_SPLIT_DATA_ON_QPS_DEFAULT;

extern int64_t NCCL_IB_TC;
extern int64_t NCCL_IB_TC_DEFAULT;

extern int64_t NCCL_IB_TIMEOUT;
extern int64_t NCCL_IB_TIMEOUT_DEFAULT;

extern int64_t NCCL_IB_USE_INLINE;
extern int64_t NCCL_IB_USE_INLINE_DEFAULT;

extern int64_t NCCL_IGNORE_CPU_AFFINITY;
extern int64_t NCCL_IGNORE_CPU_AFFINITY_DEFAULT;

extern int64_t NCCL_IGNORE_DISABLED_P2P;
extern int64_t NCCL_IGNORE_DISABLED_P2P_DEFAULT;

extern int64_t NCCL_L1_SHARED_MEMORY_CARVEOUT;
extern int64_t NCCL_L1_SHARED_MEMORY_CARVEOUT_DEFAULT;

extern std::string NCCL_LAUNCH_MODE;
extern std::string NCCL_LAUNCH_MODE_DEFAULT;

extern int64_t NCCL_LL128_BUFFSIZE;
extern int64_t NCCL_LL128_BUFFSIZE_DEFAULT;

extern int64_t NCCL_LL128_NTHREADS;
extern int64_t NCCL_LL128_NTHREADS_DEFAULT;

extern int64_t NCCL_LL_BUFFSIZE;
extern int64_t NCCL_LL_BUFFSIZE_DEFAULT;

extern int64_t NCCL_LOCAL_REGISTER;
extern int64_t NCCL_LOCAL_REGISTER_DEFAULT;

extern int64_t NCCL_MAX_CTAS;
extern int64_t NCCL_MAX_CTAS_DEFAULT;

extern int64_t NCCL_MAX_NCHANNELS;
extern int64_t NCCL_MAX_NCHANNELS_DEFAULT;

extern int64_t NCCL_MAX_NRINGS;
extern int64_t NCCL_MAX_NRINGS_DEFAULT;

extern int64_t NCCL_MAX_P2P_NCHANNELS;
extern int64_t NCCL_MAX_P2P_NCHANNELS_DEFAULT;

enum class NCCL_MEM_SYNC_DOMAIN {
  local,
  remote,
};
extern enum NCCL_MEM_SYNC_DOMAIN NCCL_MEM_SYNC_DOMAIN;
extern enum NCCL_MEM_SYNC_DOMAIN NCCL_MEM_SYNC_DOMAIN_DEFAULT;

extern int64_t NCCL_MIN_CTAS;
extern int64_t NCCL_MIN_CTAS_DEFAULT;

extern int64_t NCCL_MIN_NCHANNELS;
extern int64_t NCCL_MIN_NCHANNELS_DEFAULT;

extern int64_t NCCL_MIN_NRINGS;
extern int64_t NCCL_MIN_NRINGS_DEFAULT;

extern int64_t NCCL_MIN_P2P_NCHANNELS;
extern int64_t NCCL_MIN_P2P_NCHANNELS_DEFAULT;

extern int64_t NCCL_NCHANNELS_PER_NET_PEER;
extern int64_t NCCL_NCHANNELS_PER_NET_PEER_DEFAULT;

extern std::string NCCL_NETWORK;
extern std::string NCCL_NETWORK_DEFAULT;

extern int64_t NCCL_NET_DISABLE_INTRA;
extern int64_t NCCL_NET_DISABLE_INTRA_DEFAULT;

extern int64_t NCCL_NET_FORCE_FLUSH;
extern int64_t NCCL_NET_FORCE_FLUSH_DEFAULT;

extern std::string NCCL_NET_GDR_LEVEL;
extern std::string NCCL_NET_GDR_LEVEL_DEFAULT;

extern int64_t NCCL_NET_GDR_READ;
extern int64_t NCCL_NET_GDR_READ_DEFAULT;

extern int64_t NCCL_NET_OVERHEAD;
extern int64_t NCCL_NET_OVERHEAD_DEFAULT;

extern std::string NCCL_NET_PLUGIN;
extern std::string NCCL_NET_PLUGIN_DEFAULT;

extern int64_t NCCL_NET_SHARED_BUFFERS;
extern int64_t NCCL_NET_SHARED_BUFFERS_DEFAULT;

extern int64_t NCCL_NET_SHARED_COMMS;
extern int64_t NCCL_NET_SHARED_COMMS_DEFAULT;

extern int64_t NCCL_NSOCKS_PERTHREAD;
extern int64_t NCCL_NSOCKS_PERTHREAD_DEFAULT;

extern int64_t NCCL_NTHREADS;
extern int64_t NCCL_NTHREADS_DEFAULT;

extern int64_t NCCL_NVB_DISABLE;
extern int64_t NCCL_NVB_DISABLE_DEFAULT;

extern int64_t NCCL_NVB_PRECONNECT;
extern int64_t NCCL_NVB_PRECONNECT_DEFAULT;

extern int64_t NCCL_NVLS_ENABLE;
extern int64_t NCCL_NVLS_ENABLE_DEFAULT;

extern int64_t NCCL_NVLS_NCHANNELS;
extern int64_t NCCL_NVLS_NCHANNELS_DEFAULT;

extern int64_t NCCL_P2P_DIRECT_DISABLE;
extern int64_t NCCL_P2P_DIRECT_DISABLE_DEFAULT;

extern std::string NCCL_P2P_DISABLE;
extern std::string NCCL_P2P_DISABLE_DEFAULT;

extern std::string NCCL_P2P_LEVEL;
extern std::string NCCL_P2P_LEVEL_DEFAULT;

extern int64_t NCCL_P2P_LL_THRESHOLD;
extern int64_t NCCL_P2P_LL_THRESHOLD_DEFAULT;

extern int64_t NCCL_P2P_NET_CHUNKSIZE;
extern int64_t NCCL_P2P_NET_CHUNKSIZE_DEFAULT;

extern int64_t NCCL_P2P_NVL_CHUNKSIZE;
extern int64_t NCCL_P2P_NVL_CHUNKSIZE_DEFAULT;

extern int64_t NCCL_P2P_PCI_CHUNKSIZE;
extern int64_t NCCL_P2P_PCI_CHUNKSIZE_DEFAULT;

extern int64_t NCCL_P2P_PXN_LEVEL;
extern int64_t NCCL_P2P_PXN_LEVEL_DEFAULT;

extern int64_t NCCL_P2P_READ_ENABLE;
extern int64_t NCCL_P2P_READ_ENABLE_DEFAULT;

extern int64_t NCCL_P2P_USE_CUDA_MEMCPY;
extern int64_t NCCL_P2P_USE_CUDA_MEMCPY_DEFAULT;

extern int64_t NCCL_PROGRESS_APPENDOP_FREQ;
extern int64_t NCCL_PROGRESS_APPENDOP_FREQ_DEFAULT;

extern std::string NCCL_PROTO;
extern std::string NCCL_PROTO_DEFAULT;

extern int64_t NCCL_PROXY_APPEND_BATCH_SIZE;
extern int64_t NCCL_PROXY_APPEND_BATCH_SIZE_DEFAULT;

extern int64_t NCCL_PROXY_DUMP_SIGNAL;
extern int64_t NCCL_PROXY_DUMP_SIGNAL_DEFAULT;

extern std::string NCCL_PROXY_PROFILE;
extern std::string NCCL_PROXY_PROFILE_DEFAULT;

extern int64_t NCCL_PXN_DISABLE;
extern int64_t NCCL_PXN_DISABLE_DEFAULT;

enum class NCCL_SENDRECV_ALGO {
  orig,
  ctran,
};
extern enum NCCL_SENDRECV_ALGO NCCL_SENDRECV_ALGO;
extern enum NCCL_SENDRECV_ALGO NCCL_SENDRECV_ALGO_DEFAULT;

extern int64_t NCCL_SET_STACK_SIZE;
extern int64_t NCCL_SET_STACK_SIZE_DEFAULT;

extern int64_t NCCL_SET_THREAD_NAME;
extern int64_t NCCL_SET_THREAD_NAME_DEFAULT;

extern int64_t NCCL_SHM_DISABLE;
extern int64_t NCCL_SHM_DISABLE_DEFAULT;

extern int64_t NCCL_SHM_LOCALITY;
extern int64_t NCCL_SHM_LOCALITY_DEFAULT;

extern int64_t NCCL_SHM_MEMCPY_MODE;
extern int64_t NCCL_SHM_MEMCPY_MODE_DEFAULT;

extern int64_t NCCL_SHM_USE_CUDA_MEMCPY;
extern int64_t NCCL_SHM_USE_CUDA_MEMCPY_DEFAULT;

extern std::string NCCL_SOCKET_FAMILY;
extern std::string NCCL_SOCKET_FAMILY_DEFAULT;

extern std::string NCCL_SOCKET_IFNAME;
extern std::string NCCL_SOCKET_IFNAME_DEFAULT;

extern int64_t NCCL_SOCKET_NTHREADS;
extern int64_t NCCL_SOCKET_NTHREADS_DEFAULT;

extern std::string NCCL_THREAD_THRESHOLDS;
extern std::string NCCL_THREAD_THRESHOLDS_DEFAULT;

extern std::string NCCL_TOPO_DUMP_FILE;
extern std::string NCCL_TOPO_DUMP_FILE_DEFAULT;

extern int64_t NCCL_TOPO_DUMP_FILE_RANK;
extern int64_t NCCL_TOPO_DUMP_FILE_RANK_DEFAULT;

extern std::string NCCL_TOPO_FILE;
extern std::string NCCL_TOPO_FILE_DEFAULT;

extern std::string NCCL_TUNER_PLUGIN;
extern std::string NCCL_TUNER_PLUGIN_DEFAULT;

extern int64_t NCCL_WORK_FIFO_DEPTH;
extern int64_t NCCL_WORK_FIFO_DEPTH_DEFAULT;


void ncclCvarInit();

#endif  /* NCCL_CVARS_H_INCLUDED */
// Automatically generated by ./maint/extractcvars.py --- END
