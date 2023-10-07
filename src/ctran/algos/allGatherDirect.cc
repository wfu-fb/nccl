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
  std::vector<int> empty;
  void *sendHdl, *recvHdl;

  NCCLCHECKGOTO(comm->ctranMapper->searchRegHandle(sendbuff, sendSize, &sendHdl),
      res, fail);
  NCCLCHECKGOTO(comm->ctranMapper->searchRegHandle(recvbuff, nRanks * sendSize, &recvHdl),
      res, fail);

  g = std::unique_ptr<ctranGraph>(new ctranGraph(comm->ctranMapper));
  g->ncclKernel = reinterpret_cast<void *>(ncclKernelAllGatherCTD);

  hdl = new int[2 * nRanks];
  for (int p = 1; p < nRanks; p++) {
    int peer = (comm->rank + p) % comm->nRanks;
    NCCLCHECKGOTO(g->irecv((void *) ((uintptr_t) recvbuff + peer * sendSize),
          sendSize, peer, recvHdl, empty, &hdl[p]), res, fail);
  }
  for (int p = 0; p < nRanks; p++) {
    int peer = (comm->rank + p) % comm->nRanks;
    if (peer == comm->rank) {
      NCCLCHECKGOTO(g->icopy((void *) ((uintptr_t) recvbuff + peer * sendSize), sendbuff, sendSize, empty,
            &hdl[nRanks + p]), res, fail);
    } else {
      NCCLCHECKGOTO(g->isend(sendbuff, sendSize, peer, sendHdl, empty, &hdl[nRanks + p]), res, fail);
    }
  }
  delete[] hdl;

  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(g), stream), res, fail);

fail:
  return res;
}
