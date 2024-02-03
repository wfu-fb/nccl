// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "DdaMemHandler.h"

#include "DdaThreadedData.h"
#include "comm.h"
#include "cudawrapper.h"

namespace nccl {
namespace algorithms {

DdaMemHandler::DdaMemHandler(ncclComm_t comm) : comm_(comm) {}

DdaMemHandler::~DdaMemHandler() {
  for (const auto& kv : allMemAddrs_) {
    for (const auto& memAddr : kv.second) {
      if (memAddr.isMmapped) {
        CUDACHECKIGNORE(cudaWrapper->cudaIpcCloseMemHandle(memAddr.addr));
      }
    }
  }
}

size_t DdaMemHandler::add(void* addr) {
  LocalMemAddr memAddr;
  memAddr.addr = addr;
  memAddr.isMmapped = false;
  allMemAddrs_[comm_->rank].push_back(std::move(memAddr));
  return allMemAddrs_.at(comm_->rank).size() - 1;
}

ncclResult_t DdaMemHandler::exchangeMemHandles() {
  // data structure needs to be exchanged
  struct ExchangedHandle {
    void* addr{nullptr};
    cudaIpcMemHandle_t ipcHandle;
  };

  // prepare send/recv buffers
  const size_t kNumSendHandle = allMemAddrs_.at(comm_->rank).size();
  const size_t kNumRecvHandle = kNumSendHandle * comm_->nRanks;
  const size_t kSendSize = sizeof(ExchangedHandle) * kNumSendHandle;
  const size_t kRecvSize = kSendSize * comm_->nRanks;

  std::vector<ExchangedHandle> sendHandles(kNumSendHandle);
  std::vector<ExchangedHandle> recvHandles(kNumRecvHandle);
  ExchangedHandle* sendBuff_d{nullptr};
  ExchangedHandle* recvBuff_d{nullptr};
  CUDACHECK(cudaWrapper->cudaMalloc((void **)&sendBuff_d, kSendSize));
  CUDACHECK(cudaWrapper->cudaMalloc((void **)&recvBuff_d, kRecvSize));

  // fill up sendbuffs
  for (int i = 0; i < kNumSendHandle; ++i) {
    const auto& memAddr = allMemAddrs_.at(comm_->rank)[i];
    auto& handle = sendHandles[i];
    handle.addr = memAddr.addr;
    CUDACHECK(cudaWrapper->cudaIpcGetMemHandle(&handle.ipcHandle, memAddr.addr));
  }
  CUDACHECK(cudaWrapper->cudaMemcpy(sendBuff_d, sendHandles.data(), kSendSize, cudaMemcpyDefault));

  // exchange handles
  cudaStream_t stream;
  CUDACHECK(cudaWrapper->cudaStreamCreate(&stream));
  NCCLCHECK(ncclAllGather(
      sendBuff_d, recvBuff_d, kSendSize, ncclUint8, comm_, stream));
  CUDACHECK(cudaWrapper->cudaStreamSynchronize(stream));
  CUDACHECK(cudaWrapper->cudaStreamDestroy(stream));

  // decode received handles
  CUDACHECK(cudaWrapper->cudaMemcpy(recvHandles.data(), recvBuff_d, kRecvSize, cudaMemcpyDefault));
  for (int rank = 0; rank < comm_->nRanks; ++rank) {
    if (rank == comm_->rank) {
      // skip self
      continue;
    }

    for (int i = 0; i < kNumSendHandle; ++i) {
      const auto& handle = recvHandles[rank * kNumSendHandle + i];

      LocalMemAddr memAddr;
      if (DdaThreadedData::get()->hasRank(comm_->commHash, rank)) {
        // threaded rank, mem-addr is in my local memory space
        memAddr.addr = handle.addr;
        memAddr.isMmapped = false;
      } else {
        // rank in a different process, open Ipc
        CUDACHECK(cudaWrapper->cudaIpcOpenMemHandle(
            (void**)&memAddr.addr,
            handle.ipcHandle,
            cudaIpcMemLazyEnablePeerAccess));
        memAddr.isMmapped = true;
      }
      allMemAddrs_[rank].push_back(std::move(memAddr));
    }
  }

  CUDACHECK(cudaWrapper->cudaFree(sendBuff_d));
  CUDACHECK(cudaWrapper->cudaFree(recvBuff_d));

  return ncclSuccess;
}

void* DdaMemHandler::get(int rank, int idx) {
  auto it = allMemAddrs_.find(rank);
  if (it == allMemAddrs_.end()) {
    return nullptr;
  }

  if (idx < 0 || idx >= it->second.size()) {
    return nullptr;
  }

  return it->second.at(idx).addr;
}

} // namespace algorithms
} // namespace nccl
