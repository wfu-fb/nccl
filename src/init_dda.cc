// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <assert.h>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include "checks.h"
#include "comm.h"
#include "nccl.h"
#include "topo.h"

NCCL_PARAM(DDAAllreduceMaxTmpbufSize, "DDA_ALLREDUCE_MAX_TMPBUF_SIZE", 8 * 1024 * 1024);
NCCL_PARAM(MaxDDAThreads, "MAX_DDA_THREADS", 16);
NCCL_PARAM(ForceP2pAccess, "FORCE_P2P_ACCESS", 0);
NCCL_PARAM(DDAAllreduceLocalBufSize, "DDA_ALLREDUCE_LOCAL_BUF_SIZE", 32 * 1024 * 1024);

static std::vector<ddaMd*> ddaMdList;
static std::mutex ddaMdListMutex;

bool operator==(const ncclUniqueId& lhs, const ncclUniqueId& rhs) {
  for (int i = 0; i < sizeof(ncclUniqueId); i++) {
    if (lhs.internal[i] != rhs.internal[i]) {
      return false;
    }
  }

  return true;
}

bool operator==(const ddaMd& lhs, const ddaMd& rhs) {
  return (lhs.commId == rhs.commId);
}

static void findNvsConnectedGpus(
    struct ncclTopoNode* node,
    std::vector<int>& gpus,
    std::vector<uint64_t>& nvs) {
  nvs.push_back(node->id);
  for (int i = 0; i < node->nlinks; i++) {
    if (node->links[i].type == LINK_NVL) {
      struct ncclTopoNode* remNode = node->links[i].remNode;
      if (remNode->type == GPU) {
        gpus.push_back(i);
      } else if (remNode->type == NVS) {
        auto it = std::find(nvs.begin(), nvs.end(), remNode->id);
        if (it == nvs.end()) {
          findNvsConnectedGpus(remNode, gpus, nvs);
        }
      }
    }
  }
}

/* This function only returns two types of cliques: fully connected
 * (NVSwitch) or HCM, to support typically GPU topologies.  In all
 * other cases, we return an empty vector.
 *
 * We do not currently account for the number of NVLinks connecting
 * two GPUs because our algorithm does not take advantage of that
 * (yet!).
 *
 * This also enables p2p access between all GPUs that are connected
 * using NVLink.  Unfortunately, we do not have a good way to detect
 * which GPUs are being used for this particular job, so we enable p2p
 * access for all connected GPUs.
 */
static ncclResult_t topoDetect(
    ncclComm_t comm,
    std::vector<std::vector<int>>& cliques) {
  int nGPUs = comm->topo->nodes[GPU].count;
  uint8_t adjacencyMatrix[nGPUs][nGPUs];

  /* clear the cliques before we start */
  cliques.clear();

  /* set adjacency matrix for self as connected */
  for (int i = 0; i < nGPUs; i++) {
    for (int j = 0; j < nGPUs; j++) {
      if (i == j) {
        adjacencyMatrix[i][j] = 1;
      } else if (ncclParamForceP2pAccess()) {
        adjacencyMatrix[i][j] = 1;
      } else {
        adjacencyMatrix[i][j] = 0;
      }
    }
  }

  /* for each GPU in the system */
  for (int i = 0; i < nGPUs; i++) {
    /* for each NVLink connection on that GPU */
    for (int j = 0; j < comm->topo->nodes[GPU].nodes[i].nlinks; j++) {
      if (comm->topo->nodes[GPU].nodes[i].links[j].type == LINK_NVL) {
        struct ncclTopoNode* remNode =
            comm->topo->nodes[GPU].nodes[i].links[j].remNode;
        if (remNode->type == GPU) { /* if it is connected to a GPU */
          adjacencyMatrix[i][remNode->gpu.dev] = 1;
        } else if (remNode->type == NVS) { /* if it is connected to an NVSwitch
                                            */
          std::vector<uint64_t> nvs;
          std::vector<int> gpus;
          findNvsConnectedGpus(remNode, gpus, nvs);
          for (auto it : gpus) {
            adjacencyMatrix[i][it] = 1;
          }
        }
      }
    }
  }

  /***** Detect fully connected (NVSwitch) topology *****/
  bool connected;
  connected = true;
  for (int i = 0; i < nGPUs; i++) {
    for (int j = 0; j < nGPUs; j++) {
      if (adjacencyMatrix[i][j] == 0) {
        connected = false;
        break;
      }
    }
    if (connected == false) {
      break;
    }
  }
  if (connected) {
    std::vector<int> v;
    for (int i = 0; i < nGPUs; i++) {
      v.push_back(i);
    }
    cliques.push_back(v);
    goto exit;
  }

  /***** Detect HCM topology *****/
  for (int i = 0; i < nGPUs; i++) {
    cliques.push_back(std::vector<int>{i});
  }

  /* find cliques of size nGPUs/2 */
  for (int k = 2; k <= nGPUs / 2; k++) {
    std::vector<std::vector<int>> tmp;
    for (auto v : cliques) {
      for (int i = v.back() + 1; i < nGPUs; i++) {
        bool connected = true;
        for (auto j : v) {
          if (adjacencyMatrix[i][j] == 0) {
            connected = false;
            break;
          }
        }
        if (connected) {
          std::vector<int> w = v;
          w.push_back(i);
          tmp.push_back(w);
        }
      }
    }

    cliques.clear();
    cliques = tmp;
    tmp.clear();
  }

  /* HCM has two cliques */
  if (cliques.size() != 2) {
    goto topo_not_found;
  }

  /* In HCM, each GPU is connected to (nGPUs/2 + 1) other GPUs */
  for (int i = 0; i < nGPUs; i++) {
    int count = 0;
    for (int j = 0; j < nGPUs; j++) {
      if (adjacencyMatrix[i][j]) {
        count++;
      }
    }
    if (count != (1 + nGPUs / 2)) {
      goto topo_not_found;
    }
  }

  /* layout the cliques, so the two HCM cliques are ordered so that
   * the i'th GPU in one clique is connected to the i'th GPU in the
   * other clique.  If this is not possible, it's not HCM. */
  {
    std::vector<int> front = cliques.front();
    std::vector<int> back = cliques.back();
    std::vector<int> tmp;
    for (auto f : front) {
      /* each element in front should be connected to exactly
       * one element in back */
      for (auto b : back) {
        if (adjacencyMatrix[f][b]) {
          tmp.push_back(b);
          break;
        }
      }
    }
    assert(tmp.size() == nGPUs / 2);
    cliques.pop_back();
    cliques.push_back(tmp);
  }

exit:
  for (int i = 0; i < nGPUs; i++) {
    int dev;
    CUDACHECK(cudaGetDevice(&dev));
    if (i == dev) {
      continue;
    } else if (adjacencyMatrix[comm->cudaDev][i] == 1) {
      cudaError_t e = cudaDeviceEnablePeerAccess(i, 0);
      if (e != cudaErrorPeerAccessAlreadyEnabled && e != cudaSuccess) {
        CUDACHECK(e);
      }
    }
  }
  return ncclSuccess;

topo_not_found:
  cliques.clear();
  return ncclSuccess;
}

/*
 * This is our core function to detect the number of ranks in the same
 * virtual address space (i.e., threads).  The first communicator
 * handle that is created with a particular context ID creates and
 * enqueues an ddaMd object in the ddaMdList
 * queue.  If a new communicator handle is created with the same
 * context ID, it would point to the same ddaMd object.  The
 * number of communicator handles pointing to the ddaMd
 * object determines the number of dda ranks in this address
 * space.
 */
ncclResult_t allocDDAMd(ncclComm_t comm, ncclUniqueId commId) {
  // set enableIpc flag
  char* allreduceAlgoStr = getenv("NCCL_ALLREDUCE_ALGO");
  const bool enableIpc =
      (allreduceAlgoStr != nullptr &&
       !strcmp(allreduceAlgoStr, "dda_ipc"));

  ddaMd* md;
  ncclResult_t ret = ncclSuccess;
  std::vector<std::vector<int>> gpuCliques;

  ddaMdListMutex.lock();

  // IPC states start
  // TODO: variables should be declared close to their usage, but can't be
  // done here due to NCCLCHECKGOTO, need a fix later
  const size_t kNumHandles = 4;
  cudaIpcMemHandle_t localHdls[kNumHandles];
  void* handleSendBuf{nullptr};
  void* handleRecvBuf{nullptr};
  const size_t kHandleSize = sizeof(cudaIpcMemHandle_t);
  const size_t kBarrierSize =
      2 * comm->nRanks * comm->nRanks * sizeof(uintptr_t);
  cudaStream_t stream;
  cudaIpcMemHandle_t allHdls[comm->nRanks * kNumHandles];
  // IPC states end

  NCCLCHECKGOTO(topoDetect(comm, gpuCliques), ret, exit);

  /* allocate the ddaMd structure or find an existing
   * one for this commId */
  md = nullptr;
  for (auto t : ddaMdList) {
    if (t->commId == commId) {
      md = t;
      break;
    }
  }
  if (md == nullptr) {
    md = new ddaMd(commId, gpuCliques, enableIpc);
    ddaMdList.push_back(md);
  }

  md->insertRank(comm->rank, comm->cudaDev);

  comm->dda.md = md;
  comm->dda.barrierFlag = 0;
  comm->dda.barrierMboxId = 1;
  comm->dda.localMboxId = 1;
  CUDACHECK(
      cudaGetDeviceProperties(&comm->dda.devProp, comm->cudaDev));

  md->refCount++;

  if (md->enableIpc()) {
    // allocate dev mem
    CUDACHECK(cudaMalloc(&md->barrierMbox[0], kBarrierSize));
    CUDACHECK(cudaMalloc(&md->barrierMbox[1], kBarrierSize));
    CUDACHECK(cudaMalloc(
        &md->localSendBuff, ncclParamDDAAllreduceLocalBufSize()));
    CUDACHECK(cudaMalloc(&md->allSendBuffs, comm->nRanks * sizeof(uintptr_t)));
    CUDACHECK(cudaMalloc(
        &md->localTmpBuff, ncclParamDDAAllreduceMaxTmpbufSize()));
    CUDACHECK(cudaMalloc(&md->allTmpBuffs, comm->nRanks * sizeof(uintptr_t)));

    // allocate host mem
    md->allSendBuffsHost =
        static_cast<void**>(malloc(comm->nRanks * sizeof(uintptr_t)));
    md->allTmpBuffsHost =
        static_cast<void**>(malloc(comm->nRanks * sizeof(uintptr_t)));
    md->nRanks = comm->nRanks;

    // open local handles
    CUDACHECK(cudaIpcGetMemHandle(&localHdls[0], md->barrierMbox[0]));
    CUDACHECK(cudaIpcGetMemHandle(&localHdls[1], md->barrierMbox[1]));
    CUDACHECK(cudaIpcGetMemHandle(&localHdls[2], md->localSendBuff));
    CUDACHECK(cudaIpcGetMemHandle(&localHdls[3], md->localTmpBuff));

    // copy handles to local sendBuf
    CUDACHECK(cudaMalloc(&handleSendBuf, kHandleSize * kNumHandles));
    CUDACHECK(
        cudaMalloc(&handleRecvBuf, kHandleSize * comm->nRanks * kNumHandles));
    CUDACHECK(cudaMemcpy(
        handleSendBuf,
        &localHdls,
        kHandleSize * kNumHandles,
        cudaMemcpyDefault));

    // all gather local handles
    cudaStreamCreate(&stream);
    NCCLCHECK(ncclAllGather(
        handleSendBuf,
        handleRecvBuf,
        kHandleSize * kNumHandles,
        ncclUint8,
        comm,
        stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    // deserialize all hanles
    CUDACHECK(cudaMemcpy(
        allHdls,
        handleRecvBuf,
        kHandleSize * comm->nRanks * kNumHandles,
        cudaMemcpyDefault));

    // all gather completed, free send/recv buf
    CUDACHECK(cudaFree(handleSendBuf));
    CUDACHECK(cudaFree(handleRecvBuf));

    // update md->allSend/TmpBufs[nRanks]
    for (size_t rankIdx = 0; rankIdx < comm->nRanks; ++rankIdx) {
      const auto& barrierHdl0 = allHdls[rankIdx * kNumHandles];
      const auto& barrierHdl1 = allHdls[rankIdx * kNumHandles + 1];
      const auto& sendBufHdl = allHdls[rankIdx * kNumHandles + 2];
      const auto& tmpBufHdl = allHdls[rankIdx * kNumHandles + 3];
      if (comm->rank == rankIdx) {
        // local rank should point to local buf
        md->allSendBuffsHost[rankIdx] = md->localSendBuff;
        md->allTmpBuffsHost[rankIdx] = md->localTmpBuff;
      } else {
        // otherwise, open IPC handle
        void* remoteBuf = nullptr;
        CUDACHECK(cudaIpcOpenMemHandle(
            (void**)&remoteBuf, sendBufHdl, cudaIpcMemLazyEnablePeerAccess));
        md->allSendBuffsHost[rankIdx] = remoteBuf;

        CUDACHECK(cudaIpcOpenMemHandle(
            (void**)&remoteBuf, tmpBufHdl, cudaIpcMemLazyEnablePeerAccess));
        md->allTmpBuffsHost[rankIdx] = remoteBuf;
      }
    }
    CUDACHECK(cudaMemcpy(
        md->allSendBuffs,
        md->allSendBuffsHost,
        comm->nRanks * sizeof(uintptr_t),
        cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(
        md->allTmpBuffs,
        md->allTmpBuffsHost,
        comm->nRanks * sizeof(uintptr_t),
        cudaMemcpyDefault));

    // update md->barrierMbox, all ranks should use rank0's barrier
    // TODO should use lowest rank ID instead?
    if (comm->rank != 0) {
      void* remoteBuf = nullptr;
      CUDACHECK(cudaIpcOpenMemHandle(
          (void**)&remoteBuf, allHdls[0], cudaIpcMemLazyEnablePeerAccess));
      md->barrierMbox[0] = reinterpret_cast<uintptr_t*>(remoteBuf);

      remoteBuf = nullptr;
      CUDACHECK(cudaIpcOpenMemHandle(
          (void**)&remoteBuf, allHdls[1], cudaIpcMemLazyEnablePeerAccess));
      md->barrierMbox[1] = reinterpret_cast<uintptr_t*>(remoteBuf);
    }
  }
exit:
  ddaMdListMutex.unlock();
  return ret;
}

/* This function decreases the refCount for the ddaMd object
 * (one of the communicator pointing to it is getting freed).  If the
 * refCount reaches zero, that means no communicators are pointing to
 * it -- in that case, we can remove it from the
 * ddaMdList. */
ncclResult_t freeDDAMd(ddaMd* md, int rank) {
  ddaMdListMutex.lock();

  md->refCount--;

  if (md->refCount == 0) {
    if (md->enableIpc()) {
      // close ipc handles
      for (int i = 0; i < md->nRanks; ++i) {
        if (i == rank) {
          continue;
        }
        CUDACHECKIGNORE(cudaIpcCloseMemHandle(md->allSendBuffsHost[i]));
      }
      // free host/dev memories
      CUDACHECKIGNORE(cudaFree(md->barrierMbox[0]));
      CUDACHECKIGNORE(cudaFree(md->barrierMbox[1]));
      CUDACHECKIGNORE(cudaFree(md->localSendBuff));
      CUDACHECKIGNORE(cudaFree(md->allSendBuffs));
      CUDACHECKIGNORE(cudaFree(md->localTmpBuff));
      CUDACHECKIGNORE(cudaFree(md->allTmpBuffs));
    }

    auto mdIdx =
        std::remove(ddaMdList.begin(), ddaMdList.end(), md);
    ddaMdList.erase(mdIdx, ddaMdList.end());
    delete md;
  }

  ddaMdListMutex.unlock();

  return ncclSuccess;
}
