// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranAlgos.h"
#include "comm.h"

static ncclResult_t sendImpl(struct collOp *op) {
  ncclResult_t res = ncclSuccess;
  size_t sendSize = op->send.count * ncclTypeSize(op->send.datatype);
  void *remoteRecvBuff;
  struct ctranMapperRemoteAccessKey remoteAccessKey;
  ctranMapper *mapper = op->send.comm->ctranMapper;
  void *sendHdl;
  ctranMapperRequest *req;
  bool isComplete;

  NCCLCHECKGOTO(mapper->searchRegHandle(op->send.sendbuff, sendSize, &sendHdl),
      res, exit);

  NCCLCHECKGOTO(mapper->irecvCtrl(&remoteRecvBuff, &remoteAccessKey, op->send.peerRank,
        &req), res, exit);
  isComplete = false;
  while (isComplete == false) {
    NCCLCHECKGOTO(req->test(&isComplete), res, exit);
  }

  NCCLCHECKGOTO(mapper->iput(op->send.sendbuff, remoteRecvBuff, sendSize, op->send.peerRank,
        sendHdl, remoteAccessKey, true, &req), res, exit);
  isComplete = false;
  while (isComplete == false) {
    NCCLCHECKGOTO(req->test(&isComplete), res, exit);
  }

exit:
  return res;
}

ncclResult_t ctranSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  std::unique_ptr<struct collOp> op;

  op = std::unique_ptr<struct collOp>(new struct collOp);
  op->func = sendImpl;
  op->ncclKernel = reinterpret_cast<void *>(ncclKernelSend);
  op->send.sendbuff = sendbuff;
  op->send.count = count;
  op->send.datatype = datatype;
  op->send.peerRank = peer;
  op->send.comm = comm;

  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(op), stream), res, fail);

fail:
  return res;
}

static ncclResult_t recvImpl(struct collOp *op) {
  ncclResult_t res = ncclSuccess;
  size_t recvSize = op->recv.count * ncclTypeSize(op->recv.datatype);
  ctranMapper *mapper = op->recv.comm->ctranMapper;
  void *recvHdl;
  ctranMapperRequest *req;
  bool isComplete;
  bool notify;

  NCCLCHECKGOTO(mapper->searchRegHandle(op->recv.recvbuff, recvSize, &recvHdl),
      res, exit);

  NCCLCHECKGOTO(mapper->isendCtrl(op->recv.recvbuff, recvHdl, op->recv.peerRank, &req), res, exit);
  isComplete = false;
  while (isComplete == false) {
    NCCLCHECKGOTO(req->test(&isComplete), res, exit);
  }

  notify = false;
  while (notify == false) {
    NCCLCHECKGOTO(mapper->checkNotify(op->recv.peerRank, &notify), res, exit);
  }

exit:
  return res;
}

ncclResult_t ctranRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  std::unique_ptr<struct collOp> op;

  op = std::unique_ptr<struct collOp>(new struct collOp);
  op->func = recvImpl;
  op->ncclKernel = reinterpret_cast<void *>(ncclKernelRecv);
  op->recv.recvbuff = recvbuff;
  op->recv.count = count;
  op->recv.datatype = datatype;
  op->recv.peerRank = peer;
  op->recv.comm = comm;

  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(op), stream), res, fail);

fail:
  return res;
}
