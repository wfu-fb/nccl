// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include "comm.h"
#include "ctranAlgos.h"

static ncclResult_t impl(std::vector<std::unique_ptr<struct collOp>> opGroup) {
  ncclResult_t res = ncclSuccess;
  struct collOp* op = opGroup.front().get();
  size_t sendSize =
      op->allgather.sendcount * ncclTypeSize(op->allgather.datatype);
  int rank = op->comm->rank;
  int nRanks = op->comm->nRanks;
  int nSteps = log2i(nRanks);
  void* sendbuff = (void*)op->allgather.sendbuff;
  void* recvbuff = (void*)op->allgather.recvbuff;

  ctranMapper* mapper = op->comm->ctranMapper;
  void *sendHdl, *recvHdl;
  std::vector<size_t> peers(nSteps);
  std::vector<size_t> dists(nSteps);
  std::vector<void*> remoteRecvBuffs(nSteps);
  std::vector<struct ctranMapperRemoteAccessKey> remoteAccessKeys(nSteps);
  std::vector<ctranMapperRequest*> irecvReq(nSteps);
  std::vector<ctranMapperRequest*> isendReq(nSteps);
  std::vector<ctranMapperRequest*> iputReq(nSteps);
  ctranMapperTimestamp timestamp("ctranAllGatherRecDbl");

  bool localRegSend{false}, localRegRecv{false};

  // Calculate distance and peer per step
  for (size_t i = 0; i < nSteps; i++) {
    dists[i] = nRanks / (2 << i);
    size_t pos = (rank / dists[i]) % 2;
    peers[i] = pos == 0 ? rank + dists[i] : rank - dists[i];
  }

  NCCLCHECKGOTO(
      mapper->searchRegHandle(sendbuff, sendSize, &sendHdl), res, exit);
  if (sendHdl == nullptr) {
    NCCLCHECKGOTO(
        mapper->regMem(op->allgather.sendbuff, sendSize, &sendHdl), res, exit);
    localRegSend = true;
  }
  NCCLCHECKGOTO(
      mapper->searchRegHandle(recvbuff, nRanks * sendSize, &recvHdl),
      res,
      exit);
  if (recvHdl == nullptr) {
    NCCLCHECKGOTO(
        mapper->regMem(op->allgather.recvbuff, nRanks * sendSize, &recvHdl),
        res,
        exit);
    localRegRecv = true;
  }

  // Exchange memory handles with relevant peerse
  for (size_t i = 0; i < nSteps; i++) {
    size_t peer = peers[i];

    NCCLCHECKGOTO(
        mapper->irecvCtrl(
            &remoteRecvBuffs[i], &remoteAccessKeys[i], peer, &irecvReq[i]),
        res,
        exit);

    NCCLCHECKGOTO(
        mapper->isendCtrl(recvbuff, recvHdl, peer, &isendReq[i]), res, exit);
  }

  for (size_t i = 0; i < nSteps; i++) {
    auto peer = peers[i];

    // Block until we have handle for this peer
    NCCLCHECKGOTO(irecvReq[i]->wait(), res, exit);
    timestamp.recvCtrl.push_back(ctranMapperTimestampPoint(peer));

    for (size_t j = 0; j < (1 << i); j++) {
      size_t putOffset = j * (nRanks / (1 << i)) + rank % (nRanks / (1 << i));
      char* putFrom;
      void* putFromHdl;
      // Only need to block on the final put
      bool notify = j == (1 << i) - 1;
      ctranMapperRequest** putReqPtr = notify ? &iputReq[i] : nullptr;

      if (putOffset == rank) {
        putFrom = (char*)sendbuff;
        putFromHdl = sendHdl;
      } else {
        putFrom = (char*)recvbuff + putOffset * sendSize;
        putFromHdl = recvHdl;
      }

      NCCLCHECKGOTO(
          mapper->iput(
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
      timestamp.putIssued.push_back(ctranMapperTimestampPoint(peer));
    }
    // Wait for signal from receives
    NCCLCHECKGOTO(iputReq[i]->wait(), res, exit);
    timestamp.putComplete.push_back(ctranMapperTimestampPoint(peer));
    NCCLCHECKGOTO(mapper->waitNotify(peer), res, exit);
  }

  for (int i = 0; i < nSteps; i++) {
    NCCLCHECKGOTO(isendReq[i]->wait(), res, exit);
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

ncclResult_t ctranAllGatherRd(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
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
  NCCLCHECKGOTO(
      comm->ctranGpe->submit(
          std::move(opGroup),
          impl,
          reinterpret_cast<void*>(ncclKernelAllGatherCtranRecDbl)),
      res,
      fail);

fail:
  return res;
}
