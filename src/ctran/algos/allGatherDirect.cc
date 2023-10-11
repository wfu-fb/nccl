// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranAlgos.h"
#include "comm.h"
#include <iostream>

static ncclResult_t impl(struct collOp *op) {
  ncclResult_t res = ncclSuccess;
  size_t sendSize = op->allgather.sendcount * ncclTypeSize(op->allgather.datatype);
  int rank = op->allgather.comm->rank;
  int nRanks = op->allgather.comm->nRanks;
  ctranMapper *mapper = op->allgather.comm->ctranMapper;
  void *sendHdl, *recvHdl;
  void *remoteRecvBuffs[nRanks];
  struct ctranMapperRemoteAccessKey remoteAccessKeys[nRanks];
  ctranMapperRequest *irecvReq[nRanks];
  ctranMapperRequest *isendReq[nRanks];
  ctranMapperRequest *iputReq[nRanks];
  bool irecvComplete[nRanks];
  bool isendComplete[nRanks];
  bool iputComplete[nRanks];
  bool localRegSend, localRegRecv;

  for (int i = 0; i < nRanks; i++) {
    irecvComplete[i] = false;
    isendComplete[i] = false;
    iputComplete[i] = false;
  }

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

  for (int p = 1; p < nRanks; p++) {
    int peer = (rank + p) % nRanks;
    NCCLCHECKGOTO(mapper->irecvCtrl(&remoteRecvBuffs[peer],
          &remoteAccessKeys[peer], peer, &irecvReq[peer]), res, exit);

    NCCLCHECKGOTO(mapper->isendCtrl(op->allgather.recvbuff,
          recvHdl, peer, &isendReq[peer]), res, exit);
  }

  irecvComplete[rank] = true;
  isendComplete[rank] = true;

  if ((uintptr_t) op->allgather.recvbuff + rank * sendSize != (uintptr_t) op->allgather.sendbuff) {
    NCCLCHECKGOTO(mapper->icopy((void *) ((uintptr_t) op->allgather.recvbuff + rank * sendSize),
          op->allgather.sendbuff, sendSize, &iputReq[rank]), res, exit);
  } else {
    iputComplete[rank] = true;
  }

  bool pendingRecv;
  do {
    pendingRecv = false;
    for (int p = 1; p < nRanks; p++) {
      int peer = (rank + p) % nRanks;
      if (irecvComplete[peer] == true) {
        continue;
      }

      NCCLCHECKGOTO(irecvReq[peer]->test(&irecvComplete[peer]), res, exit);
      if (irecvComplete[peer] == false) {
        pendingRecv = true;
        continue;
      }

      NCCLCHECKGOTO(mapper->iput(op->allgather.sendbuff,
            (void *) ((uintptr_t) remoteRecvBuffs[peer] + rank * sendSize), sendSize, peer,
            sendHdl, remoteAccessKeys[peer], true, &iputReq[peer]), res, exit);
    }
  } while (pendingRecv == true);

  for (int p = 1; p < nRanks; p++) {
    int peer = (rank + p) % nRanks;
    if (isendComplete[peer] == false) {
      NCCLCHECKGOTO(isendReq[peer]->wait(), res, exit);
    }
    if (iputComplete[peer] == false) {
      NCCLCHECKGOTO(iputReq[peer]->wait(), res, exit);
    }
    NCCLCHECKGOTO(mapper->waitNotify(peer), res, exit);
  }

  if (iputComplete[rank] == false) {
    NCCLCHECKGOTO(iputReq[rank]->wait(), res, exit);
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

ncclResult_t ctranAllGatherDirect(const void* sendbuff, void* recvbuff,
	size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  std::unique_ptr<struct collOp> op;

  op = std::unique_ptr<struct collOp>(new struct collOp);
  op->func = impl;
  op->ncclKernel = reinterpret_cast<void *>(ncclKernelAllGatherCTD);
  op->allgather.sendbuff = sendbuff;
  op->allgather.recvbuff = recvbuff;
  op->allgather.sendcount = sendcount;
  op->allgather.datatype = datatype;
  op->allgather.comm = comm;

  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(op), stream), res, fail);

fail:
  return res;
}
