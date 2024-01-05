// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <nccl.h>
#include "Ctran.h"
#include "comm.h"
#include <deque>

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_CTRAN_RING_STEP
   type        : uint64_t
   default     : 4194304
   description : |-
     Pipeline step size for the CTRAN ring algorithm.

 - name        : NCCL_CTRAN_RING_MAX_OUTSTANDING
   type        : int
   default     : 8
   description : |-
     Max number of outstanding puts in the ring pipeline.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

struct PutQElem {
  char* lAddr;
  char* rAddr;
  size_t size;
  void* hdl;
};

static ncclResult_t impl(std::vector<std::unique_ptr<struct OpElem>> opGroup) {
  ncclResult_t res = ncclSuccess;
  struct OpElem* op = opGroup.front().get();
  size_t sendSize =
      op->allgather.sendcount * ncclTypeSize(op->allgather.datatype);
  ncclComm_t comm = opGroup.front()->comm;
  int rank = op->comm->rank;
  int nRanks = op->comm->nRanks;
  void *sendHdl, *recvHdl;
  bool localRegSend, localRegRecv;
  void* remoteRecvBuff;
  struct CtranMapperRemoteAccessKey remoteAccessKey;
  CtranMapper *mapper = comm->ctran->mapper.get();

  if (nRanks == 1) {
    return ncclSuccess;
  }

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("CtranAllGatherRing"));

  CtranMapperRequest* irecvReq;
  CtranMapperRequest* isendReq;
  int left = (rank + nRanks - 1) % nRanks;
  int right = (rank + 1) % nRanks;

  size_t stepSize = std::min(NCCL_CTRAN_RING_STEP, sendSize);
  size_t stepsPerBlock = std::max(1LU, (sendSize + stepSize - 1) / stepSize); // ceilDiv
  size_t maxOutstandingPuts = NCCL_CTRAN_RING_MAX_OUTSTANDING;
  std::deque<PutQElem> putQ;
  std::deque<CtranMapperRequest*> iputReqs;
  uint64_t blockNum{0};
  uint64_t stepInBlock{0};

  NCCLCHECKGOTO(
      mapper->searchRegHandle(
          op->allgather.sendbuff, sendSize, &sendHdl, &localRegSend),
      res,
      exit);

  NCCLCHECKGOTO(
      mapper->searchRegHandle(
          op->allgather.recvbuff, nRanks * sendSize, &recvHdl, &localRegRecv),
      res,
      exit);

  NCCLCHECKGOTO(
      mapper->irecvCtrl(
          &remoteRecvBuff, &remoteAccessKey, right, &irecvReq),
      res,
      exit);
  NCCLCHECKGOTO(
      mapper->isendCtrl(
          op->allgather.recvbuff, recvHdl, left, &isendReq),
      res,
      exit);
  NCCLCHECKGOTO(irecvReq->wait(), res, exit);
  timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(right));

  // Push addresses for first block onto deque
  for (int i=0; i<stepsPerBlock; ++i) {
    char* lAddr = (char*)op->allgather.sendbuff + i * stepSize;
    char* rAddr = (char*)remoteRecvBuff + rank * sendSize + i * stepSize;
    size_t size = std::min(stepSize, sendSize - i * stepSize);
    putQ.push_back({lAddr, rAddr, size, sendHdl});
  }

  while (!putQ.empty() || !iputReqs.empty() || (blockNum < nRanks-1)) {
    // Check for notifications from left and queue up corresponding sends
    while (true) {
      bool notifyRcvd{false};
      NCCLCHECKGOTO(mapper->checkNotify(left, &notifyRcvd), res, exit);
      if (!notifyRcvd) {
        break;
      }
      // Don't queue send for final step
      if (blockNum < nRanks - 2) {
        int blockId = (rank - blockNum - 1 + nRanks) % nRanks;
        char* lAddr = (char*)op->allgather.recvbuff + blockId * sendSize + stepInBlock * stepSize;
        char* rAddr = (char*)remoteRecvBuff + blockId * sendSize + stepInBlock * stepSize;
        size_t size = std::min(stepSize, sendSize - stepInBlock * stepSize);
        putQ.push_back({lAddr, rAddr, size, recvHdl});
      }
      if (stepInBlock == stepsPerBlock - 1) {
        ++blockNum;
      }
      stepInBlock = (stepInBlock + 1) % stepsPerBlock;
    }

    // Remove any completed puts from putQ, making room for new puts if possible
    while (!iputReqs.empty()) {
      bool done;
      NCCLCHECKGOTO(iputReqs.front()->test(&done), res, exit);
      if (done) {
        iputReqs.pop_front();
      } else {
        break;
      }
    }

    // Issue new puts if we're ready to, up to max outstanding
    while (!putQ.empty() && iputReqs.size() < maxOutstandingPuts) {
      CtranMapperRequest *req;
      const auto& e = putQ.front();
      // Always notify receiver and always get a cqe back
      NCCLCHECKGOTO(
        mapper->iput(
          e.lAddr, e.rAddr, e.size, right, e.hdl, remoteAccessKey, true, &req),
        res,
        exit);
      iputReqs.push_back(req);
      putQ.pop_front();
    }
  }

  NCCLCHECKGOTO(isendReq->wait(), res, exit);

  if (localRegSend == true) {
    NCCLCHECKGOTO(mapper->deregMem(sendHdl), res, exit);
  }
  if (localRegRecv == true) {
    NCCLCHECKGOTO(mapper->deregMem(recvHdl), res, exit);
  }

  mapper->timestamps.push_back(std::move(timestamp));
  mapper->reportProfiling();

exit:
  return res;
}

ncclResult_t ctranAllGatherRing(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  std::unique_ptr<struct OpElem> op;

  CTRAN_COLL_INFO("ctranAllGatherRing", sendbuff, recvbuff, sendcount, datatype, -1, comm, stream);

  /* copy data for out-of-place allgather */
  if ((uintptr_t)recvbuff + comm->rank * sendcount * ncclTypeSize(datatype) !=
      (uintptr_t)sendbuff) {
    CtranMapperRequest* req;
    comm->ctran->mapper->icopy(
        (void*)((uintptr_t)recvbuff + comm->rank * sendcount * ncclTypeSize(datatype)),
        sendbuff,
        sendcount * ncclTypeSize(datatype),
        stream,
        &req);
  }

  op = std::unique_ptr<struct OpElem>(
      new OpElem(OpElem::opType::ALLGATHER, stream, comm));
  op->allgather.sendbuff = sendbuff;
  op->allgather.recvbuff = recvbuff;
  op->allgather.sendcount = sendcount;
  op->allgather.datatype = datatype;

  opGroup.push_back(std::move(op));
  NCCLCHECKGOTO(
      comm->ctran->gpe->submit(
          std::move(opGroup),
          impl,
          reinterpret_cast<void*>(ncclKernelAllGatherCtranRing)),
      res,
      fail);

fail:
  return res;
}
