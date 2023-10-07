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
  std::vector<int> empty;
  void *sendHdl, *recvHdl;
  int left = (comm->rank + comm->nRanks - 1) % comm->nRanks;
  int right = (comm->rank + 1) % comm->nRanks;

  NCCLCHECKGOTO(comm->ctranMapper->searchRegHandle(sendbuff, sendcount * ncclTypeSize(datatype), &sendHdl),
      res, fail);
  NCCLCHECKGOTO(comm->ctranMapper->searchRegHandle(recvbuff, nRanks * sendcount * ncclTypeSize(datatype), &recvHdl),
      res, fail);

  g = std::unique_ptr<ctranGraph>(new ctranGraph(comm->ctranMapper));
  g->ncclKernel = reinterpret_cast<void *>(ncclKernelAllGatherCTR);

  hdl = new int[2 * nRanks];
  NCCLCHECK(g->icopy((void *) ((uintptr_t) recvbuff + comm->rank * sendSize),
      sendbuff, sendSize, empty, &hdl[0]));
  for (int p = 1; p < nRanks; p++) {
    int peer = (comm->rank - p + nRanks) % nRanks;
    NCCLCHECK(g->irecv((void *) ((uintptr_t) recvbuff + peer * sendSize),
          sendSize, left, recvHdl, empty, &hdl[p]));
  }
  for (int p = 1; p < nRanks; p++) {
    int peer = (comm->rank - p + 1 + nRanks) % nRanks;
    if (p == 1) {
      NCCLCHECK(g->isend(sendbuff, sendSize, right, sendHdl, empty, &hdl[nRanks + p]));
    } else {
      std::vector<int> v;
      v.push_back(hdl[p-1]);
      NCCLCHECK(g->isend((void *) ((uintptr_t) recvbuff + peer * sendSize),
            sendSize, right, recvHdl, v, &hdl[nRanks + p]));
      v.clear();
    }
  }
  delete[] hdl;

  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(g), stream), res, fail);

fail:
  return res;
}
