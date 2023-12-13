// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include "Ctran.h"
#include "comm.h"

static ncclResult_t impl(std::vector<std::unique_ptr<struct OpElem>> opGroup) {
  ncclResult_t res = ncclSuccess;
  struct OpElem* op = opGroup.front().get();
  size_t sendSize =
      op->allgather.sendcount * ncclTypeSize(op->allgather.datatype);
  ncclComm_t comm = opGroup.front()->comm;
  int rank = op->comm->rank;
  int nRanks = op->comm->nRanks;
  int nSteps = log2i(nRanks);
  void* sendbuff = (void*)op->allgather.sendbuff;
  void* recvbuff = (void*)op->allgather.recvbuff;

  void *sendHdl, *recvHdl;
  std::vector<size_t> peers(nSteps);
  std::vector<size_t> dists(nSteps);
  std::vector<void*> remoteRecvBuffs(nSteps);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(nSteps);
  std::vector<CtranMapperRequest*> irecvReq(nSteps);
  std::vector<CtranMapperRequest*> isendReq(nSteps);
  std::vector<CtranMapperRequest*> iputReq(nSteps);
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("CtranAllGatherRecDbl"));

  bool localRegSend{false}, localRegRecv{false};

  // Calculate distance and peer per step
  for (size_t i = 0; i < nSteps; i++) {
    dists[i] = nRanks / (2 << i);
    size_t pos = (rank / dists[i]) % 2;
    peers[i] = pos == 0 ? rank + dists[i] : rank - dists[i];
  }

  NCCLCHECKGOTO(
      comm->ctran->mapper->searchRegHandle(
          sendbuff, sendSize, &sendHdl, &localRegSend),
      res,
      exit);
  NCCLCHECKGOTO(
      comm->ctran->mapper->searchRegHandle(
          recvbuff, nRanks * sendSize, &recvHdl, &localRegRecv),
      res,
      exit);

  // Exchange memory handles with relevant peerse
  for (size_t i = 0; i < nSteps; i++) {
    size_t peer = peers[i];

    NCCLCHECKGOTO(
        comm->ctran->mapper->irecvCtrl(
            &remoteRecvBuffs[i], &remoteAccessKeys[i], peer, &irecvReq[i]),
        res,
        exit);

    NCCLCHECKGOTO(
        comm->ctran->mapper->isendCtrl(recvbuff, recvHdl, peer, &isendReq[i]),
        res,
        exit);
  }

  for (size_t i = 0; i < nSteps; i++) {
    auto peer = peers[i];

    // Block until we have handle for this peer
    NCCLCHECKGOTO(irecvReq[i]->wait(), res, exit);
    timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(peer));

    for (size_t j = 0; j < (1 << i); j++) {
      size_t putOffset = j * (nRanks / (1 << i)) + rank % (nRanks / (1 << i));
      char* putFrom;
      void* putFromHdl;
      // Only need to block on the final put
      bool notify = j == (1 << i) - 1;
      CtranMapperRequest** putReqPtr = notify ? &iputReq[i] : nullptr;

      if (putOffset == rank) {
        putFrom = (char*)sendbuff;
        putFromHdl = sendHdl;
      } else {
        putFrom = (char*)recvbuff + putOffset * sendSize;
        putFromHdl = recvHdl;
      }

      NCCLCHECKGOTO(
          comm->ctran->mapper->iput(
              putFrom,
              (char*)remoteRecvBuffs[i] + putOffset * sendSize,
              sendSize,
              peer,
              putFromHdl,
              remoteAccessKeys[i],
              notify,
              putReqPtr),
          res,
          exit);
      timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));
    }
    // Wait for signal from receives
    NCCLCHECKGOTO(iputReq[i]->wait(), res, exit);
    timestamp->putComplete.push_back(CtranMapperTimestampPoint(peer));
    NCCLCHECKGOTO(comm->ctran->mapper->waitNotify(peer), res, exit);
  }

  for (int i = 0; i < nSteps; i++) {
    NCCLCHECKGOTO(isendReq[i]->wait(), res, exit);
  }

  if (localRegSend == true) {
    NCCLCHECKGOTO(comm->ctran->mapper->deregMem(sendHdl), res, exit);
  }
  if (localRegRecv == true) {
    NCCLCHECKGOTO(comm->ctran->mapper->deregMem(recvHdl), res, exit);
  }

  comm->ctran->mapper->timestamps.push_back(std::move(timestamp));
  comm->ctran->mapper->reportProfiling();

exit:
  return res;
}

ncclResult_t ctranAllGatherRd(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  CTRAN_COLL_INFO(
      "CtranAllGatherRd",
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      -1,
      comm,
      stream);

  ncclResult_t res = ncclSuccess;
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  std::unique_ptr<struct OpElem> op;

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

  op = std::unique_ptr<struct OpElem>(new struct OpElem);
  op->type = OpElem::opType::ALLGATHER;
  op->comm = comm;
  op->stream = stream;
  op->allgather.sendbuff = sendbuff;
  op->allgather.recvbuff = recvbuff;
  op->allgather.sendcount = sendcount;
  op->allgather.datatype = datatype;

  opGroup.push_back(std::move(op));
  NCCLCHECKGOTO(
      comm->ctran->gpe->submit(
          std::move(opGroup),
          impl,
          reinterpret_cast<void*>(ncclKernelAllGatherCtranRecDbl)),
      res,
      fail);

fail:
  return res;
}
