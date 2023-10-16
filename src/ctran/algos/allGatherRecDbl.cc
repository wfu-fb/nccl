// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include "comm.h"
#include "ctranAlgos.h"

static ncclResult_t impl(std::vector<std::unique_ptr<struct collOp>> opGroup) {
  ncclResult_t res = ncclSuccess;
  struct collOp *op = opGroup.front().get();
  size_t sendSize =
      op->allgather.sendcount * ncclTypeSize(op->allgather.datatype);
  int rank = op->comm->rank;
  int nRanks = op->comm->nRanks;
  int nSteps = log2i(nRanks);
  void* sendbuff = (void*)op->allgather.sendbuff;
  void* recvbuff = (void*)op->allgather.recvbuff;
  bool inPlace = (char*)sendbuff == (char*)recvbuff + rank * sendSize;

  ctranMapper* mapper = op->comm->ctranMapper;
  void *sendHdl, *recvHdl;
  std::unique_ptr<size_t[]> peers = std::unique_ptr<size_t[]>(new size_t[nSteps]);
  std::unique_ptr<size_t[]> dists = std::unique_ptr<size_t[]>(new size_t[nSteps]);
  std::unique_ptr<void *[]> remoteRecvBuffs = std::unique_ptr<void *[]>(new void *[nSteps]);
  std::unique_ptr<struct ctranMapperRemoteAccessKey[]> remoteAccessKeys =
    std::unique_ptr<struct ctranMapperRemoteAccessKey[]>(new struct ctranMapperRemoteAccessKey[nSteps]);
  std::unique_ptr<ctranMapperRequest *[]> irecvReq =
    std::unique_ptr<ctranMapperRequest *[]>(new ctranMapperRequest *[nSteps]);
  std::unique_ptr<ctranMapperRequest *[]> isendReq =
    std::unique_ptr<ctranMapperRequest *[]>(new ctranMapperRequest *[nSteps]);
  std::unique_ptr<ctranMapperRequest *[]> iputReq =
    std::unique_ptr<ctranMapperRequest *[]>(new ctranMapperRequest *[nSteps]);

  ctranMapperRequest* copyReq;
  bool copyComplete = inPlace;

  // Calculate distance and peer per step
  for (size_t i = 0; i < nSteps; i++) {
    dists[i] = nRanks / (2 << i);
    size_t pos = (rank / dists[i]) % 2;
    peers[i] = pos == 0 ? rank + dists[i] : rank - dists[i];
  }

  NCCLCHECKGOTO(
      mapper->searchRegHandle(sendbuff, sendSize, &sendHdl), res, exit);
  NCCLCHECKGOTO(
      mapper->searchRegHandle(recvbuff, nRanks * sendSize, &recvHdl),
      res,
      exit);

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

  if (!inPlace) {
    NCCLCHECKGOTO(
        mapper->icopy(
            (char*)recvbuff + rank * sendSize, sendbuff, sendSize, &copyReq),
        res,
        exit);
  }

  for (size_t i = 0; i < nSteps; i++) {
    auto peer = peers[i];

    // Block until we have handle for this peer
    NCCLCHECKGOTO(irecvReq[i]->wait(), res, exit);

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
    }
    // Wait for signal from receives
    NCCLCHECKGOTO(iputReq[i]->wait(), res, exit);
    NCCLCHECKGOTO(mapper->waitNotify(peer), res, exit);
  }

  for (int i=0; i<nSteps; i++){
    NCCLCHECKGOTO(isendReq[i]->wait(), res, exit);
  }

  if (!copyComplete) {
    NCCLCHECKGOTO(copyReq->wait(), res, exit);
  }

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
  std::unique_ptr<struct collOp> op;

  op = std::unique_ptr<struct collOp>(new struct collOp);
  op->type = collOp::opType::ALLGATHER;
  op->comm = comm;
  op->stream = stream;
  op->allgather.sendbuff = sendbuff;
  op->allgather.recvbuff = recvbuff;
  op->allgather.sendcount = sendcount;
  op->allgather.datatype = datatype;

  std::vector<std::unique_ptr<struct collOp>> opGroup;
  opGroup.push_back(std::move(op));
  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(opGroup), impl,
        reinterpret_cast<void *>(ncclKernelAllGatherCtranRecDbl)), res, fail);

fail:
  return res;
}
