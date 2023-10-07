// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranAlgos.h"
#include "comm.h"

ncclResult_t ctranAllGatherRing(const void* sendbuff, void* recvbuff,
	size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  int nRanks = comm->nRanks;
  size_t sendSize = sendcount * ncclTypeSize(datatype);
  std::unique_ptr<ctranGraph> g;
  int *hdl;
  int cpyHdl;
  std::vector<int> deps;
  void *sendHdl, *recvHdl;
  void *isendbuf = (void*) sendbuff;
  void *irecvbuf = recvbuff;
  int left = (comm->rank + comm->nRanks - 1) % comm->nRanks;
  int right = (comm->rank + 1) % comm->nRanks;

  NCCLCHECKGOTO(comm->ctranMapper->searchRegHandle(sendbuff, sendcount * ncclTypeSize(datatype), &sendHdl),
      res, fail);
  NCCLCHECKGOTO(comm->ctranMapper->searchRegHandle(recvbuff, nRanks * sendcount * ncclTypeSize(datatype), &recvHdl),
      res, fail);

  g = std::unique_ptr<ctranGraph>(new ctranGraph(comm->ctranMapper, "ncclAllGatherCtranRing"));
  g->ncclKernel = reinterpret_cast<void *>(ncclKernelAllGatherCTR);

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
  NCCLCHECK(g->icopy((void *) ((uintptr_t) recvbuff + comm->rank * sendSize),
      sendbuff, sendSize, deps, &hdl[0]));
  for (int p = 1; p < nRanks; p++) {
    int peer = (comm->rank - p + nRanks) % nRanks;
    NCCLCHECK(g->irecv((void *) ((uintptr_t) irecvbuf + peer * sendSize),
          sendSize, left, recvHdl, deps, &hdl[p]));
    if (irecvbuf != recvbuff) {
      int dummy;
      NCCLCHECKGOTO(g->icopy((void *) ((uintptr_t) recvbuff + peer * sendSize),
                             (void *) ((uintptr_t) irecvbuf + peer * sendSize),
                             sendSize, {hdl[p]}, &dummy), res, fail);
    }
  }
  for (int p = 1; p < nRanks; p++) {
    int peer = (comm->rank - p + 1 + nRanks) % nRanks;
    if (p == 1) {
      NCCLCHECK(g->isend(isendbuf, sendSize, right, sendHdl, deps, &hdl[nRanks + p]));
    } else {
      std::vector<int> v;
      v.push_back(hdl[p-1]);
      NCCLCHECK(g->isend((void *) ((uintptr_t) irecvbuf + peer * sendSize),
            sendSize, right, recvHdl, v, &hdl[nRanks + p]));
      v.clear();
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
