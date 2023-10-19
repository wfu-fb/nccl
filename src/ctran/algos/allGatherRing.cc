// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranAlgos.h"
#include "comm.h"

static ncclResult_t impl(std::vector<std::unique_ptr<struct collOp>> opGroup) {
  ncclResult_t res = ncclSuccess;
  struct collOp *op = opGroup.front().get();
  size_t sendSize = op->allgather.sendcount * ncclTypeSize(op->allgather.datatype);
  int rank = op->comm->rank;
  int nRanks = op->comm->nRanks;
  ctranMapper *mapper = op->comm->ctranMapper;
  void *sendHdl, *recvHdl;
  bool localRegSend, localRegRecv;
  void *remoteRecvBuff;
  struct ctranMapperRemoteAccessKey remoteAccessKey;

  ctranMapperRequest *irecvReq;
  ctranMapperRequest *isendReq;
  ctranMapperRequest *iputReq;
  bool iputComplete;
  int left = (rank + nRanks - 1) % nRanks;
  int right = (rank + 1) % nRanks;

  NCCLCHECKGOTO(mapper->searchRegHandle(op->allgather.sendbuff, sendSize, &sendHdl),
      res, exit);
  if (sendHdl == nullptr) {
    NCCLCHECKGOTO(mapper->regMem(op->allgather.sendbuff, sendSize, &sendHdl), res, exit);
    localRegSend = true;
  } else {
    localRegSend = false;
  }

  NCCLCHECKGOTO(mapper->searchRegHandle(op->allgather.recvbuff,
        nRanks * sendSize, &recvHdl), res, exit);
  if (recvHdl == nullptr) {
    NCCLCHECKGOTO(mapper->regMem(op->allgather.recvbuff, nRanks * sendSize, &recvHdl), res, exit);
    localRegRecv = true;
  } else {
    localRegRecv = false;
  }

  NCCLCHECKGOTO(mapper->irecvCtrl(&remoteRecvBuff, &remoteAccessKey, right, &irecvReq), res, exit);
  NCCLCHECKGOTO(mapper->isendCtrl(op->allgather.recvbuff, recvHdl, left, &isendReq), res, exit);
  NCCLCHECKGOTO(irecvReq->wait(), res, exit);

  NCCLCHECKGOTO(mapper->iput(op->allgather.sendbuff,
        (void *) ((uintptr_t) remoteRecvBuff + rank * sendSize), sendSize, right,
        sendHdl, remoteAccessKey, true, (nRanks > 2) ? nullptr : &iputReq), res, exit);

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
  NCCLCHECKGOTO(isendReq->wait(), res, exit);

  if (iputComplete == false) {
    NCCLCHECKGOTO(iputReq->wait(), res, exit);
  }

  if (localRegSend == true) {
    NCCLCHECKGOTO(mapper->deregMem(sendHdl), res, exit);
  }
  if (localRegRecv == true) {
    NCCLCHECKGOTO(mapper->deregMem(recvHdl), res, exit);
  }

exit:
  return res;
}

ncclResult_t ctranAllGatherRing(const void* sendbuff, void* recvbuff,
	size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  std::vector<std::unique_ptr<struct collOp>> opGroup;
  std::unique_ptr<struct collOp> op;

  /* copy data for out-of-place allgather */
  if ((uintptr_t)recvbuff + comm->rank * sendcount * ncclTypeSize(datatype) !=
      (uintptr_t)sendbuff) {
    opGroup.push_back(createCpyOp(
        (void*)((uintptr_t)recvbuff + comm->rank * sendcount * ncclTypeSize(datatype)),
        sendbuff,
        sendcount * ncclTypeSize(datatype),
        comm,
        stream));
  }

  op = std::unique_ptr<struct collOp>(new struct collOp);
  op->type = collOp::opType::ALLGATHER;
  op->comm = comm;
  op->stream = stream;
  op->allgather.sendbuff = sendbuff;
  op->allgather.recvbuff = recvbuff;
  op->allgather.sendcount = sendcount;
  op->allgather.datatype = datatype;

  opGroup.push_back(std::move(op));
  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(opGroup), impl,
        reinterpret_cast<void *>(ncclKernelAllGatherCtranRing)), res, fail);

fail:
  return res;
}
