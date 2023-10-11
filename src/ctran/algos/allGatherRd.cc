// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include "comm.h"
#include "ctranAlgos.h"

ncclResult_t ctranAllGatherRd(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;

#if 0
  int rank = comm->rank;
  int nRanks = comm->nRanks;
  int nSteps = log2i(nRanks);
  size_t chunkSize = sendcount * ncclTypeSize(datatype);
  std::unique_ptr<ctranGraph> g;
  void *sendMemHdl, *recvMemHdl;
  void* irecvbuf = recvbuff;
  bool inPlace = (char*)sendbuff == (char*)recvbuff + rank * chunkSize;
  bool sendMemReg, recvMemReg;

  int depHandle;
  std::vector<int> allDeps;

  NCCLCHECKGOTO(
      comm->ctranMapper->searchRegHandle(
          sendbuff, sendcount * ncclTypeSize(datatype), &sendMemHdl),
      res,
      fail);
  NCCLCHECKGOTO(
      comm->ctranMapper->searchRegHandle(
          recvbuff, nRanks * sendcount * ncclTypeSize(datatype), &recvMemHdl),
      res,
      fail);

  sendMemReg = sendMemHdl != nullptr;
  recvMemReg = recvMemHdl != nullptr;

  if (!recvMemHdl) {
    size_t irecvBuffSize = std::max(nRanks * chunkSize, 4096LU);
    TRACE(
        "CTRAN: No registered handle found for recv buffer %p; allocating buffer of size %ul",
        recvbuff,
        irecvBuffSize);
    comm->ctranMapper->getTmpBuf(&irecvbuf, irecvBuffSize, &recvMemHdl);
    if (!recvMemHdl) {
      res = ncclInternalError;
      goto fail;
    }
    TRACE("New recvMemHdl: %p\n", recvMemHdl);
  }

  g = std::unique_ptr<ctranGraph>(
      new ctranGraph(comm->ctranMapper, "ncclAllGatherCtranRd"));
  g->ncclKernel = reinterpret_cast<void*>(ncclKernelAllGatherCTRD);

  // Copy local chunk to recvbuff if not in-place
  if (!inPlace || !sendMemReg) {
    NCCLCHECKGOTO(
        g->icopy(
            (char*)irecvbuf + rank * chunkSize,
            sendbuff,
            chunkSize,
            {},
            &depHandle),
        res,
        fail);
    if (!sendMemReg) {
      // Only need to wait for this copy if sendbuff isn't registered;
      // otherwise, we'll send directly from sendbuff
      allDeps.push_back(depHandle);
    }

    if (!recvMemReg) {
      // Also need to copy to final buffer if recvbuff wasn't registered
      // Never need to block on this
      NCCLCHECKGOTO(
          g->icopy(
              (char*)recvbuff + rank * chunkSize,
              sendbuff,
              chunkSize,
              {},
              &depHandle),
          res,
          fail);
    }
  }

  for (size_t i = 0; i < nSteps; i++) {
    size_t dist = nRanks / (2 << i);
    size_t pos = (rank / dist) % 2;
    size_t partner = pos == 0 ? rank + dist : rank - dist;
    std::vector<int> thisIterRecvs;

    for (size_t j = 0; j < (1 << i); j++) {
      { // Receive
        size_t recvOffset =
            j * (nRanks / (1 << i)) + partner % (nRanks / (1 << i));

        NCCLCHECKGOTO(
            g->irecv(
                (char*)irecvbuf + recvOffset * chunkSize,
                chunkSize,
                partner,
                recvMemHdl,
                allDeps,
                &depHandle),
            res,
            fail);
        thisIterRecvs.push_back(depHandle);

        if (!recvMemReg) {
          int dummy;
          NCCLCHECKGOTO(
              g->icopy(
                  (char*)recvbuff + recvOffset * chunkSize,
                  (char*)irecvbuf + recvOffset * chunkSize,
                  chunkSize,
                  {depHandle},
                  &dummy),
              res,
              fail);
        }
      }

      { // Send

        // If sendbuff was registered and we're sending this rank's chunk,
        // send directly from sendbuff. Otherwise, send from irecvbuf
        size_t sendOffset =
            j * (nRanks / (1 << i)) + rank % (nRanks / (1 << i));
        char* sendFrom;
        void* sendFromMemHdl;
        if ((sendOffset == rank) && sendMemReg) {
          sendFrom = (char*)sendbuff;
          sendFromMemHdl = sendMemHdl;
        } else {
          sendFrom = (char*)irecvbuf + sendOffset * chunkSize;
          sendFromMemHdl = recvMemHdl;
        }

        NCCLCHECKGOTO(
            g->isend(
                sendFrom,
                chunkSize,
                partner,
                sendFromMemHdl,
                allDeps,
                &depHandle),
            res,
            fail);
      }
    }
    allDeps.insert(allDeps.end(), thisIterRecvs.begin(), thisIterRecvs.end());
  }
  if (!recvMemReg) {
    // add callback followed by the completion of the graph
    // if tmp buffers are used, release them back to pool
    g->registerCB([comm, irecvbuf, recvMemHdl](void) -> void {
      comm->ctranMapper->releaseTmpBuf(irecvbuf, recvMemHdl);
    });
  }
  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(g), stream), res, fail);

fail:
#endif
  return res;
}
