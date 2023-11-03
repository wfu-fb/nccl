// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranAlgos.h"
#include "comm.h"
#include <iostream>

static ncclResult_t impl(std::vector<std::unique_ptr<struct collOp>> opGroup) {
  ncclResult_t res = ncclSuccess;
  struct collOp *op = opGroup.front().get();
  size_t sendSize = op->allgather.sendcount * ncclTypeSize(op->allgather.datatype);
  int rank = op->comm->rank;
  int nRanks = op->comm->nRanks;
  ctranMapper *mapper = op->comm->ctranMapper;
  void *sendHdl, *recvHdl;
  std::vector<void *> remoteRecvBuffs(nRanks);
  std::vector<struct ctranMapperRemoteAccessKey> remoteAccessKeys(nRanks);
  std::vector<ctranMapperRequest *> irecvReq(nRanks);
  std::vector<ctranMapperRequest *> isendReq(nRanks);
  std::vector<ctranMapperRequest *> iputReq(nRanks);
  std::vector<bool> irecvComplete(nRanks);
  std::vector<bool> isendComplete(nRanks);
  std::vector<bool> iputComplete(nRanks);
  bool localRegSend, localRegRecv;
  ctranMapperTimestamp timestamp("ctranAllGatherDirect");

  for (int i = 0; i < nRanks; i++) {
    irecvComplete[i] = false;
    isendComplete[i] = false;
    iputComplete[i] = false;
  }

  NCCLCHECKGOTO(mapper->searchRegHandle(op->allgather.sendbuff, sendSize, &sendHdl, &localRegSend),
      res, exit);
  NCCLCHECKGOTO(mapper->searchRegHandle(op->allgather.recvbuff,
        nRanks * sendSize, &recvHdl, &localRegRecv), res, exit);

  for (int p = 1; p < nRanks; p++) {
    int peer = (rank + p) % nRanks;
    NCCLCHECKGOTO(mapper->irecvCtrl(&remoteRecvBuffs[peer],
          &remoteAccessKeys[peer], peer, &irecvReq[peer]), res, exit);

    NCCLCHECKGOTO(mapper->isendCtrl(op->allgather.recvbuff,
          recvHdl, peer, &isendReq[peer]), res, exit);
  }

  irecvComplete[rank] = true;
  isendComplete[rank] = true;
  iputComplete[rank] = true;

  bool pendingRecv;
  do {
    pendingRecv = false;
    for (int p = 1; p < nRanks; p++) {
      int peer = (rank + p) % nRanks;
      if (irecvComplete[peer] == true) {
        continue;
      }

      bool isComplete;
      NCCLCHECKGOTO(irecvReq[peer]->test(&isComplete), res, exit);
      irecvComplete[peer] = isComplete;
      if (irecvComplete[peer] == false) {
        pendingRecv = true;
        continue;
      }

      timestamp.recvCtrl.push_back(ctranMapperTimestampPoint(peer));
      NCCLCHECKGOTO(mapper->iput(op->allgather.sendbuff,
            (void *) ((uintptr_t) remoteRecvBuffs[peer] + rank * sendSize), sendSize, peer,
            sendHdl, remoteAccessKeys[peer], true, &iputReq[peer]), res, exit);
      timestamp.putIssued.push_back(ctranMapperTimestampPoint(peer));
    }
  } while (pendingRecv == true);

  for (int p = 1; p < nRanks; p++) {
    int peer = (rank + p) % nRanks;
    if (isendComplete[peer] == false) {
      NCCLCHECKGOTO(isendReq[peer]->wait(), res, exit);
    }
    if (iputComplete[peer] == false) {
      NCCLCHECKGOTO(iputReq[peer]->wait(), res, exit);
      timestamp.putComplete.push_back(ctranMapperTimestampPoint(peer));
    }
    NCCLCHECKGOTO(mapper->waitNotify(peer), res, exit);
  }

  if (localRegSend == true) {
    NCCLCHECKGOTO(mapper->deregMem(sendHdl), res, exit);
  }
  if (localRegRecv == true) {
    NCCLCHECKGOTO(mapper->deregMem(recvHdl), res, exit);
  }

  mapper->timestamps.push_back(timestamp);

exit:
  return res;
}

ncclResult_t ctranAllGatherDirect(const void* sendbuff, void* recvbuff,
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
        reinterpret_cast<void *>(ncclKernelAllGatherCtranDirect)), res, fail);

fail:
  return res;
}
