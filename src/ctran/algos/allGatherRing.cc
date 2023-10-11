// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranAlgos.h"
#include "comm.h"

static ncclResult_t impl(struct collOp *op) {
  ncclResult_t res = ncclSuccess;
  size_t sendSize = op->allgather.sendcount * ncclTypeSize(op->allgather.datatype);
  int rank = op->allgather.comm->rank;
  int nRanks = op->allgather.comm->nRanks;
  ctranMapper *mapper = op->allgather.comm->ctranMapper;
  void *sendHdl, *recvHdl;
  void *remoteRecvBuff;
  struct ctranMapperRemoteAccessKey remoteAccessKey;

  ctranMapperRequest *irecvReq;
  ctranMapperRequest *isendReq;
  ctranMapperRequest *iputReq;
  ctranMapperRequest *icopyReq;
  bool iputComplete;
  int left = (rank + nRanks - 1) % nRanks;
  int right = (rank + 1) % nRanks;

  NCCLCHECKGOTO(mapper->searchRegHandle(op->allgather.sendbuff, sendSize, &sendHdl),
      res, exit);
  NCCLCHECKGOTO(mapper->searchRegHandle(op->allgather.recvbuff,
        nRanks * sendSize, &recvHdl), res, exit);

  NCCLCHECKGOTO(mapper->icopy((void *) ((uintptr_t) op->allgather.recvbuff + rank * sendSize),
        op->allgather.sendbuff, sendSize, &icopyReq), res, exit);

  NCCLCHECKGOTO(mapper->irecvCtrl(&remoteRecvBuff, &remoteAccessKey, right, &irecvReq), res, exit);
  NCCLCHECKGOTO(mapper->isendCtrl(op->allgather.recvbuff, recvHdl, left, &isendReq), res, exit);
  NCCLCHECKGOTO(irecvReq->wait(), res, exit);

  NCCLCHECKGOTO(mapper->iput(op->allgather.sendbuff,
        (void *) ((uintptr_t) remoteRecvBuff + rank * sendSize), sendSize, right,
        sendHdl, remoteAccessKey, true, nullptr), res, exit);

  iputComplete = true;
  for (int i = 0; i < nRanks - 2; i++) {
    int blockId = (rank - i - 1 + nRanks) % nRanks;

    NCCLCHECKGOTO(mapper->waitNotify(left), res, exit);
    NCCLCHECKGOTO(mapper->iput(
          (void *) ((uintptr_t) op->allgather.recvbuff + blockId * sendSize),
          (void *) ((uintptr_t) remoteRecvBuff + blockId * sendSize), sendSize, right,
          recvHdl, remoteAccessKey, true, (i < nRanks - 3) ? nullptr : &iputReq), res, exit);
    iputComplete = false;
  }

  NCCLCHECKGOTO(mapper->waitNotify(left), res, exit);
  NCCLCHECKGOTO(icopyReq->wait(), res, exit);
  NCCLCHECKGOTO(isendReq->wait(), res, exit);

  if (iputComplete == false) {
    NCCLCHECKGOTO(iputReq->wait(), res, exit);
  }

exit:
  return res;
}

ncclResult_t ctranAllGatherRing(const void* sendbuff, void* recvbuff,
	size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  std::unique_ptr<struct collOp> op;

  op = std::unique_ptr<struct collOp>(new struct collOp);
  op->func = impl;
  op->ncclKernel = reinterpret_cast<void *>(ncclKernelAllGatherCTR);
  op->allgather.sendbuff = sendbuff;
  op->allgather.recvbuff = recvbuff;
  op->allgather.sendcount = sendcount;
  op->allgather.datatype = datatype;
  op->allgather.comm = comm;

  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(op), stream), res, fail);

fail:
  return res;
}
