// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_COMM_DDA_H_
#define NCCL_COMM_DDA_H_

#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "checks.h"

/*
 * We maintain three classes here:
 *
 * ddaPrivateMd -- this is local to each communicator and contains
 * rank-private information.
 *
 * ddaThreadSharedMd -- this is thread-shared metadata; all threaded
 * ranks within this communicator on this node share this information.
 *
 * ddaCliqueSharedMd -- this is meta-data that is shared within each
 * clique (mesh-connected GPUs) of threaded ranks.
 */

int64_t ncclParamMaxDDAThreads(void);
int64_t ncclParamDDAAllreduceTmpbuffSize(void);

typedef enum {
  NCCL_DDA_TOPO_TYPE__NVS,
  NCCL_DDA_TOPO_TYPE__HCM,
  NCCL_DDA_TOPO_TYPE__UNKNOWN,
} ncclDDATopoType_t;

/* each clique (direct NVLink connected group) of ranks */
class ddaCliqueSharedMd {
 public:
  ddaCliqueSharedMd(std::vector<int> gpuClique) {
    this->gpus = std::move(gpuClique);

    /* mailbox for ranks to exchange their source buffer
     * information.  We create two copies and swap between the two
     * each time the collective is called.  This way, if one rank
     * is delayed in completing its work, we don't overwrite this
     * data in the next iteration. */
    for (int i = 0; i < 2; i++) {
      this->barrierMbox[i] = nullptr;
    }
  }

  ~ddaCliqueSharedMd() {
    for (int i = 0; i < 2; i++) {
      if (this->barrierMbox[i] != nullptr) {
        CUDACHECKIGNORE(cudaFree(this->barrierMbox[i]));
      }

      for (auto it : this->rankToLocalMbox[i]) {
        void* buf = it.second;
        CUDACHECKIGNORE(cudaFree(buf));
      }
      this->rankToLocalMbox[i].clear();
    }
  }

  void insertRank(int rank, int cudaDev) {
    auto it = std::find(this->gpus.begin(), this->gpus.end(), cudaDev);
    if (it == this->gpus.end()) {
      return;
    }

    if (this->rankToGpu.empty()) {
      int numBarrierPtrs = 2 * this->gpus.size();

      for (int i = 0; i < 2; i++) {
        CUDACHECKIGNORE(cudaMalloc(
            &this->barrierMbox[i], numBarrierPtrs * sizeof(uintptr_t)));
        CUDACHECKIGNORE(cudaMemset(
            this->barrierMbox[i], 0, numBarrierPtrs * sizeof(uintptr_t)));
      }
    }

    this->rankToGpu[rank] = cudaDev;

    for (int i = 0; i < 2; i++) {
      void* buf;

      CUDACHECKIGNORE(cudaMalloc(&buf, sizeof(uintptr_t)));
      CUDACHECKIGNORE(cudaMemset(buf, 0, sizeof(uintptr_t)));

      this->rankToLocalMbox[i][rank] = reinterpret_cast<uintptr_t*>(buf);
    }
  }

  /* mapping from rank to the GPU ID, temporary buffer, and local mbox */
  std::vector<int> gpus;
  std::unordered_map<int, int> rankToGpu;
  std::unordered_map<int, uintptr_t*> rankToLocalMbox[2];
  uintptr_t* barrierMbox[2];
};

/* metadata for dda ranks: contains the clique of GPUs (currently
 * all of the GPUs in the system), and a refcount of the number of
 * communicator handles in this address space that point to the same
 * commId */
class ddaThreadSharedMd {
 public:
  ddaThreadSharedMd(
      ncclUniqueId commId,
      std::vector<std::vector<int>> &gpuCliques,
      bool enableIpc = false)
      : enableIpc_(enableIpc) {
    this->commId = commId;

    for (auto const &it : gpuCliques) {
      ddaCliqueSharedMd* clique = new ddaCliqueSharedMd(it);
      this->cliques.push_back(clique);
    }

    this->refCount = 0;
  }

  ~ddaThreadSharedMd() {
    for (auto c : cliques) {
      delete c;
    }
  }

  void insertRank(int rank, int cudaDev) {
    for (auto it : cliques) {
      it->insertRank(rank, cudaDev);
    }
  }

  bool enableIpc() const {
    return enableIpc_;
  }

  ncclUniqueId commId;
  std::vector<ddaCliqueSharedMd*> cliques;
  int refCount;

 private:
  // enable IPC or not
  const bool enableIpc_{false};
};

class ddaPrivateMd {
public:
  ddaPrivateMd(ddaThreadSharedMd *threadSharedMd, int cudaDev, int numCliques) {
    this->barrierFlag = 0;
    this->barrierMboxId = 1;
    CUDACHECKIGNORE(cudaGetDeviceProperties(&this->devProp, cudaDev));

    CUDACHECKIGNORE(cudaMalloc(&this->tmpbuff, ncclParamDDAAllreduceTmpbuffSize()));

    /* add topology information */
    this->topoType = NCCL_DDA_TOPO_TYPE__UNKNOWN;
    if (numCliques == 1) {
      this->topoType = NCCL_DDA_TOPO_TYPE__NVS;
    } else if (numCliques == 2) {
      this->topoType = NCCL_DDA_TOPO_TYPE__HCM;
    }

    this->threadSharedMd = threadSharedMd;
  }

  ~ddaPrivateMd() {
    CUDACHECKIGNORE(cudaFree(this->tmpbuff));
  }

  // flag indicating that each rank has arrived at the barrier
  uintptr_t barrierFlag;

  // barrier mailbox ID to use
  int barrierMboxId;

  // device properties
  cudaDeviceProp devProp;

  // local tmpbuff
  void* tmpbuff{nullptr};

  // topology type
  ncclDDATopoType_t topoType;

  // thread-shared meta-data
  ddaThreadSharedMd *threadSharedMd;

  // ipc states
  // barrier mailboxes
  uintptr_t* barrierMbox[2];

  // all ranks' tmpbuff addresses
  void** allTmpSendbuffs{nullptr};
  // all ranks' tmpbuff host-addrs
  void** allTmpSendbuffsHost{nullptr};
};

ncclResult_t allocDDAMd(ncclComm *comm, ncclUniqueId commId);
ncclResult_t freeDDAMd(ncclComm *comm);

static inline int typeSize(ncclDataType_t datatype)
{
  switch (datatype) {
    case ncclInt8:
      return sizeof(int8_t);

    case ncclUint8:
      return sizeof(uint8_t);

    case ncclInt32:
      return sizeof(int32_t);

    case ncclUint32:
      return sizeof(uint32_t);

    case ncclInt64:
      return sizeof(int64_t);

    case ncclUint64:
      return sizeof(uint64_t);

    case ncclFloat16:
      return sizeof(half);

  case ncclFloat32:
      return sizeof(float);

    case ncclFloat64:
      return sizeof(double);

#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
      return sizeof(__nv_bfloat16);
#endif

    default:
      return 0;
  }
}

#endif
