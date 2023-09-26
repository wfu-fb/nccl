// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ddaMemHandles.h"
#include "comm.h"

ddaMemHandles::ddaMemHandles(ddaThreadSharedMd *threadSharedMd, ncclComm *comm) {
  this->threadSharedMd = threadSharedMd;
  this->comm = comm;
}

ddaMemHandles::~ddaMemHandles() {
  for (int i = 0; i < this->comm->nRanks; i++) {
    for (auto h : this->allMemHandles[i]) {
      if (h.isMmapped) {
        CUDACHECKIGNORE(cudaIpcCloseMemHandle(h.addr));
      }
    }
  }
}

ncclResult_t ddaMemHandles::insertMemHandle(void *addr, std::string name) {
  cudaIpcMemHandle_t handle;
  ncclResult_t res = ncclSuccess;
  IpcHandleState s;

  auto got = allMemHandles.find(this->comm->rank);

  CUDACHECKGOTO(cudaIpcGetMemHandle(&handle, addr), res, exit);

  s.name = name;
  s.addr = addr;
  s.handle = handle;
  s.isMmapped = false;

  if (got != allMemHandles.end()) {
    got->second.push_back(s);
  } else {
    std::vector<IpcHandleState> v;
    v.push_back(s);
    allMemHandles[this->comm->rank] = v;
  }

exit:
  return res;
}

ncclResult_t ddaMemHandles::exchangeMemHandles(void) {
  struct serializedHandle {
    char name[MEM_HANDLE_NAME_LEN_MAX];
    void *addr;
    cudaIpcMemHandle_t handle;
  };
  size_t numHandles = allMemHandles[this->comm->rank].size();
  int sendSize = numHandles * sizeof(struct serializedHandle);
  int recvSize = sendSize * this->comm->nRanks;
  struct serializedHandle *serializedLocalHandles;
  struct serializedHandle *serializedLocalHandlesHost;
  struct serializedHandle *serializedGlobalHandles;
  struct serializedHandle *serializedGlobalHandlesHost;
  int idx;
  cudaStream_t s;
  ncclResult_t res = ncclSuccess;

  CUDACHECKGOTO(cudaMalloc(&serializedLocalHandles, sendSize), res, exit);
  CUDACHECKGOTO(cudaMalloc(&serializedGlobalHandles, recvSize), res, exit);
  CUDACHECKGOTO(cudaHostAlloc(&serializedLocalHandlesHost, sendSize, cudaHostAllocDefault), res, exit);
  CUDACHECKGOTO(cudaHostAlloc(&serializedGlobalHandlesHost, recvSize, cudaHostAllocDefault), res, exit);

  /* serialize local handles */
  idx = 0;
  for (auto h : allMemHandles[this->comm->rank]) {
    if (h.name.length() > MEM_HANDLE_NAME_LEN_MAX) {
      WARN("DDA: mem Handle %s length is longer than the allocated bytes (%d)",
          h.name.c_str(), MEM_HANDLE_NAME_LEN_MAX);
      res = ncclSystemError;
      goto exit;
    }
    strcpy(serializedLocalHandlesHost[idx].name, h.name.c_str());
    serializedLocalHandlesHost[idx].addr = h.addr;
    serializedLocalHandlesHost[idx].handle = h.handle;
    idx++;
  }
  CUDACHECKGOTO(cudaMemcpy(serializedLocalHandles, serializedLocalHandlesHost,
                           sendSize, cudaMemcpyDefault), res, exit);

  CUDACHECKGOTO(cudaStreamCreate(&s), res, exit);
  NCCLCHECK(ncclAllGather(serializedLocalHandles, serializedGlobalHandles, sendSize, ncclUint8,
                          this->comm, s));
  CUDACHECKGOTO(cudaStreamSynchronize(s), res, exit);
  CUDACHECKGOTO(cudaStreamDestroy(s), res, exit);

  CUDACHECKGOTO(cudaMemcpy(serializedGlobalHandlesHost, serializedGlobalHandles,
                           recvSize, cudaMemcpyDefault), res, exit);

  /* deserialize global handles */
  for (int r = 0; r < this->comm->nRanks; r++) {
    if (r == this->comm->rank) {
      continue;
    }
    std::vector<IpcHandleState> v;
    for (int n = 0; n < numHandles; n++) {
      struct serializedHandle *h = serializedGlobalHandlesHost + r * numHandles + n;

      if (this->threadSharedMd->searchRank(r)) {
        IpcHandleState s;
        s.name = static_cast<std::string>(h->name);
        s.addr = h->addr;
        s.handle = h->handle;
        s.isMmapped = false;
        v.push_back(s);
      } else {
        IpcHandleState s;
        void *localPtr;
        CUDACHECKGOTO(cudaIpcOpenMemHandle((void**)&localPtr, h->handle,
                                           cudaIpcMemLazyEnablePeerAccess), res, exit);
        s.name = static_cast<std::string>(h->name);
        s.addr = localPtr;
        s.handle = h->handle;
        s.isMmapped = true;
        v.push_back(s);
      }
    }
    allMemHandles[r] = v;
  }

  CUDACHECKGOTO(cudaFree(serializedLocalHandles), res, exit);
  CUDACHECKGOTO(cudaFree(serializedGlobalHandles), res, exit);
  CUDACHECKGOTO(cudaFreeHost(serializedLocalHandlesHost), res, exit);
  CUDACHECKGOTO(cudaFreeHost(serializedGlobalHandlesHost), res, exit);

exit:
  return res;
}

void *ddaMemHandles::getMemAddr(int rank, std::string name) {
  for (auto t : allMemHandles[rank]) {
    if (t.name == name) {
      return t.addr;
    }
  }
  return nullptr;
}
