// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "Ctran.h"
#include "comm.h"
#include <iostream>

static ncclResult_t impl(std::vector<std::unique_ptr<struct OpElem>> opGroup) {
  ncclResult_t res = ncclSuccess;
  struct OpElem *op = opGroup.front().get();
  size_t sendSize = op->allgather.sendcount * ncclTypeSize(op->allgather.datatype);
  ncclComm_t comm = opGroup.front()->comm;
  int rank = op->comm->rank;
  int nRanks = op->comm->nRanks;
  void *sendHdl, *recvHdl;
  std::vector<void *> remoteRecvBuffs(nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(nRanks);
  std::vector<CtranMapperRequest *> irecvReq(nRanks);
  std::vector<CtranMapperRequest *> isendReq(nRanks);
  std::vector<CtranMapperRequest *> iputReq(nRanks);
  std::vector<bool> irecvComplete(nRanks);
  std::vector<bool> isendComplete(nRanks);
  std::vector<bool> iputComplete(nRanks);
  bool localRegSend, localRegRecv;
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("CtranAllgatherDirect"));

  for (int i = 0; i < nRanks; i++) {
    irecvComplete[i] = false;
    isendComplete[i] = false;
    iputComplete[i] = false;
  }

  NCCLCHECKGOTO(comm->ctran->mapper->searchRegHandle(op->allgather.sendbuff, sendSize, &sendHdl, &localRegSend),
      res, exit);
  NCCLCHECKGOTO(comm->ctran->mapper->searchRegHandle(op->allgather.recvbuff,
        nRanks * sendSize, &recvHdl, &localRegRecv), res, exit);

  for (int p = 1; p < nRanks; p++) {
    int peer = (rank + p) % nRanks;
    NCCLCHECKGOTO(comm->ctran->mapper->irecvCtrl(&remoteRecvBuffs[peer],
          &remoteAccessKeys[peer], peer, &irecvReq[peer]), res, exit);

    NCCLCHECKGOTO(comm->ctran->mapper->isendCtrl(op->allgather.recvbuff,
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

      timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(peer));
      NCCLCHECKGOTO(comm->ctran->mapper->iput(op->allgather.sendbuff,
            (void *) ((uintptr_t) remoteRecvBuffs[peer] + rank * sendSize), sendSize, peer,
            sendHdl, remoteAccessKeys[peer], true, &iputReq[peer]), res, exit);
      timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));
    }
  } while (pendingRecv == true);

  for (int p = 1; p < nRanks; p++) {
    int peer = (rank + p) % nRanks;
    if (isendComplete[peer] == false) {
      NCCLCHECKGOTO(isendReq[peer]->wait(), res, exit);
    }
    if (iputComplete[peer] == false) {
      NCCLCHECKGOTO(iputReq[peer]->wait(), res, exit);
      timestamp->putComplete.push_back(CtranMapperTimestampPoint(peer));
    }
    NCCLCHECKGOTO(comm->ctran->mapper->waitNotify(peer), res, exit);
  }

  if (localRegSend == true) {
    NCCLCHECKGOTO(comm->ctran->mapper->deregMem(sendHdl), res, exit);
  }
  if (localRegRecv == true) {
    NCCLCHECKGOTO(comm->ctran->mapper->deregMem(recvHdl), res, exit);
  }

  comm->ctran->mapper->timestamps.emplace_back(std::move(timestamp));
  comm->ctran->mapper->reportProfiling();

exit:
  return res;
}

ncclResult_t ctranAllGatherDirect(const void* sendbuff, void* recvbuff,
	size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  CTRAN_COLL_INFO("CtranAllGatherDirect", sendbuff, recvbuff, sendcount, datatype, -1, comm, stream);

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  std::unique_ptr<struct OpElem> op;

  /* copy data for out-of-place allgather */
  if ((uintptr_t)recvbuff + comm->rank * sendcount * ncclTypeSize(datatype) !=
      (uintptr_t)sendbuff) {
    CtranMapperRequest *req;
    comm->ctran->mapper->icopy(
        (void*)((uintptr_t)recvbuff + comm->rank * sendcount * ncclTypeSize(datatype)),
        sendbuff,
        sendcount * ncclTypeSize(datatype),
        stream,
        &req);
  }

  op = std::unique_ptr<struct OpElem>(
      new OpElem(OpElem::opType::ALLGATHER, comm));
  op->allgather.sendbuff = sendbuff;
  op->allgather.recvbuff = recvbuff;
  op->allgather.sendcount = sendcount;
  op->allgather.datatype = datatype;
  opGroup.push_back(std::move(op));

  auto config = KernelConfig(KernelConfig::KernelType::ALLGATHER, stream);
  // kernel arguments are unused for now; needed for NVL path support
  ctranKernelSetAllGatherArgs(
      sendbuff,
      recvbuff,
      sendcount * ncclTypeSize(datatype),
      comm->ctran->algo->devState_d,
      &config.args);

  NCCLCHECK(comm->ctran->gpe->submit(
      std::move(opGroup),
      impl,
      config,
      reinterpret_cast<void*>(ncclKernelAllGatherCtranDirect)));

  return ncclSuccess;
}
