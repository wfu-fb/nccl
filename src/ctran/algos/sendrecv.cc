// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranAlgos.h"
#include "comm.h"

static ncclResult_t sendImpl(std::vector<std::unique_ptr<struct collOp>> opGroup) {
  ncclResult_t res = ncclSuccess;
  struct collOp *op = opGroup.front().get();
  size_t sendSize = op->send.count * ncclTypeSize(op->send.datatype);
  void *remoteRecvBuff;
  struct ctranMapperRemoteAccessKey remoteAccessKey;
  ctranMapper *mapper = op->comm->ctranMapper;
  void *sendHdl;
  ctranMapperRequest *req;
  bool isComplete;
  bool localRegSend;

  NCCLCHECKGOTO(mapper->searchRegHandle(op->send.sendbuff, sendSize, &sendHdl),
      res, exit);
  if (sendHdl == nullptr) {
    NCCLCHECKGOTO(mapper->regMem(op->send.sendbuff, sendSize, &sendHdl), res, exit);
    localRegSend = true;
  } else {
    localRegSend = false;
  }

  NCCLCHECKGOTO(mapper->irecvCtrl(&remoteRecvBuff, &remoteAccessKey, op->send.peerRank,
        &req), res, exit);
  NCCLCHECKGOTO(req->wait(), res, exit);

  NCCLCHECKGOTO(mapper->iput(op->send.sendbuff, remoteRecvBuff, sendSize, op->send.peerRank,
        sendHdl, remoteAccessKey, true, &req), res, exit);
  NCCLCHECKGOTO(req->wait(), res, exit);

  if (localRegSend == true) {
    NCCLCHECKGOTO(mapper->deregMem(sendHdl), res, exit);
  }

exit:
  return res;
}

ncclResult_t ctranSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  std::unique_ptr<struct collOp> op;

  op = std::unique_ptr<struct collOp>(new struct collOp);
  op->type = collOp::opType::SEND;
  op->comm = comm;
  op->stream = stream;
  op->send.sendbuff = sendbuff;
  op->send.count = count;
  op->send.datatype = datatype;
  op->send.peerRank = peer;

  std::vector<std::unique_ptr<struct collOp>> opGroup;
  opGroup.push_back(std::move(op));
  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(opGroup), sendImpl,
        reinterpret_cast<void *>(ncclKernelSend)), res, fail);

fail:
  return res;
}

static ncclResult_t recvImpl(std::vector<std::unique_ptr<struct collOp>> opGroup) {
  ncclResult_t res = ncclSuccess;
  struct collOp *op = opGroup.front().get();
  size_t recvSize = op->recv.count * ncclTypeSize(op->recv.datatype);
  ctranMapper *mapper = op->comm->ctranMapper;
  void *recvHdl;
  ctranMapperRequest *req;
  bool isComplete;
  bool notify;
  bool localRegRecv;

  NCCLCHECKGOTO(mapper->searchRegHandle(op->recv.recvbuff, recvSize, &recvHdl),
      res, exit);
  if (recvHdl == nullptr) {
    NCCLCHECKGOTO(mapper->regMem(op->recv.recvbuff, recvSize, &recvHdl), res, exit);
    localRegRecv = true;
  } else {
    localRegRecv = false;
  }

  NCCLCHECKGOTO(mapper->isendCtrl(op->recv.recvbuff, recvHdl, op->recv.peerRank, &req), res, exit);
  NCCLCHECKGOTO(req->wait(), res, exit);
  NCCLCHECKGOTO(mapper->waitNotify(op->recv.peerRank), res, exit);

  if (localRegRecv == true) {
    NCCLCHECKGOTO(mapper->deregMem(recvHdl), res, exit);
  }

exit:
  return res;
}

ncclResult_t ctranRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  std::unique_ptr<struct collOp> op;

  op = std::unique_ptr<struct collOp>(new struct collOp);
  op->type = collOp::opType::RECV;
  op->comm = comm;
  op->stream = stream;
  op->recv.recvbuff = recvbuff;
  op->recv.count = count;
  op->recv.datatype = datatype;
  op->recv.peerRank = peer;

  std::vector<std::unique_ptr<struct collOp>> opGroup;
  opGroup.push_back(std::move(op));
  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(opGroup), recvImpl,
        reinterpret_cast<void *>(ncclKernelRecv)), res, fail);

fail:
  return res;
}
