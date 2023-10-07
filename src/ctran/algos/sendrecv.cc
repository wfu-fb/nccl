// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranAlgos.h"
#include "comm.h"

ncclResult_t ctranSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  std::unique_ptr<ctranGraph> g;
  std::vector<int> empty;
  void *sendHdl;
  size_t sendSize = count * ncclTypeSize(datatype);

  NCCLCHECKGOTO(comm->ctranMapper->searchRegHandle(sendbuff, sendSize, &sendHdl),
      res, fail);

  g = std::unique_ptr<ctranGraph>(new ctranGraph(comm->ctranMapper, "ncclSendCtran"));
  g->ncclKernel = reinterpret_cast<void *>(ncclKernelSend);

  int dummy;
  NCCLCHECKGOTO(g->isend(sendbuff, sendSize, peer, sendHdl, empty, &dummy),
      res, fail);

  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(g), stream), res, fail);

fail:
  return res;
}

ncclResult_t ctranRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  std::unique_ptr<ctranGraph> g;
  std::vector<int> empty;
  void *recvHdl;
  size_t recvSize = count * ncclTypeSize(datatype);

  NCCLCHECKGOTO(comm->ctranMapper->searchRegHandle(recvbuff, recvSize, &recvHdl),
      res, fail);

  g = std::unique_ptr<ctranGraph>(new ctranGraph(comm->ctranMapper, "ncclRecvCtran"));
  g->ncclKernel = reinterpret_cast<void *>(ncclKernelRecv);

  int dummy;
  NCCLCHECKGOTO(g->irecv(recvbuff, recvSize, peer, recvHdl, empty, &dummy), res, fail);

  NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(g), stream), res, fail);

fail:
  return res;
}
