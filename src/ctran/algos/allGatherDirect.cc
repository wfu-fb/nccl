// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranAlgos.h"
#include "comm.h"

ncclResult_t ctranAllGatherDirect(const void* sendbuff, void* recvbuff,
	size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  int nRanks = comm->nRanks;
  size_t sendSize = sendcount * ncclTypeSize(datatype);
  std::unique_ptr<ctranGraph> g;
  int *hdl;
  int cpyHdl = 0;
  std::vector<int> deps;
  void *sendHdl, *recvHdl;
  void *isendbuf = (void*) sendbuff;
  void *irecvbuf = recvbuff;

  NCCLCHECKGOTO(comm->ctranMapper->searchRegHandle(sendbuff, sendSize, &sendHdl),
      res, fail);
  NCCLCHECKGOTO(comm->ctranMapper->searchRegHandle(recvbuff, nRanks * sendSize, &recvHdl),
      res, fail);

  g = std::unique_ptr<ctranGraph>(new ctranGraph(comm->ctranMapper, "ncclAllGatherCtranDirect"));
  g->ncclKernel = reinterpret_cast<void *>(ncclKernelAllGatherCTD);

  if (!sendHdl) {
    TRACE("CTRAN: No registered handle found for send buffer %p", sendbuff);
    comm->ctranMapper->getTmpBuf(&isendbuf, sendSize, &sendHdl);
    /* copy data from user buf to tmp buf */
    NCCLCHECKGOTO(g->icopy(isendbuf, sendbuff, sendSize, {}, &cpyHdl), res, fail);
    deps.push_back(cpyHdl);
  }
  if (!recvHdl) {
    TRACE("CTRAN: No registered handle found for recv buffer %p", recvbuff);
    comm->ctranMapper->getTmpBuf(&irecvbuf, nRanks * sendSize, &recvHdl);
  }

  hdl = new int[2 * nRanks];
  for (int p = 1; p < nRanks; p++) {
    int peer = (comm->rank + p) % comm->nRanks;
    NCCLCHECKGOTO(g->irecv((void *) ((uintptr_t) irecvbuf + peer * sendSize),
          sendSize, peer, recvHdl, deps, &hdl[p]), res, fail);
    if (irecvbuf != recvbuff) {
      int dummy;
      NCCLCHECKGOTO(g->icopy((void *) ((uintptr_t) recvbuff + peer * sendSize),
                             (void *) ((uintptr_t) irecvbuf + peer * sendSize),
                             sendSize, {hdl[p]}, &dummy), res, fail);
    }
  }
  for (int p = 0; p < nRanks; p++) {
    int peer = (comm->rank + p) % comm->nRanks;
    if (peer == comm->rank) {
      NCCLCHECKGOTO(g->icopy((void *) ((uintptr_t) recvbuff + peer * sendSize), isendbuf, sendSize, deps,
            &hdl[nRanks + p]), res, fail);
    } else {
      NCCLCHECKGOTO(g->isend(isendbuf, sendSize, peer, sendHdl, deps, &hdl[nRanks + p]), res, fail);
    }
  }

  delete[] hdl;

  // add callback followed by the completion of the graph
  g->registerCB([comm, irecvbuf, recvbuff, recvHdl, isendbuf, sendbuff, sendHdl](void) -> void {
    // if tmp buffers are used, release them back to pool
    if (irecvbuf != recvbuff) {
      comm->ctranMapper->releaseTmpBuf(irecvbuf, recvHdl);
    }
    if (isendbuf != sendbuff) {
      comm->ctranMapper->releaseTmpBuf(isendbuf, sendHdl);
    }
  });

  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(g), stream), res, fail);

fail:
  return res;
}
