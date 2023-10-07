// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include "ctranMapper.h"
#include "ctranMapperImpl.h"
#include "comm.h"

ctranMapperShared::ctranMapperShared() {
  this->id = 0;
}

// FIXME: This is a poor-man's attempt at getting a uniqueId for the
// communicator.  Unfortunately, ncclUniqueId is not really unique (child
// communicators share it), and commHash is not collision-free so
// multiple communicators can have the same hash.  I suspect we might
// need a full MPI-like contextId detection method; this is a hack for
// now.  -- Pavan Balaji (9/5/2023)
ncclResult_t ctranMapperShared::getUniqueId(ncclComm *comm, uint64_t *id) {
  ncclResult_t res = ncclSuccess;

  this->m.lock();

  uint64_t *devBufSrc, *devBufDst;
  CUDACHECKGOTO(cudaMalloc(&devBufSrc, sizeof(uint64_t)), res, exit);
  CUDACHECKGOTO(cudaMalloc(&devBufDst, sizeof(uint64_t)), res, exit);
  CUDACHECKGOTO(cudaMemcpy(devBufSrc, &this->id, sizeof(uint64_t), cudaMemcpyDefault), res, exit);

  cudaStream_t s;
  CUDACHECKGOTO(cudaStreamCreate(&s), res, exit);
  NCCLCHECKGOTO(ncclAllReduce(devBufSrc, devBufDst, 1, ncclUint64, ncclMax, comm, s), res, exit);
  CUDACHECKGOTO(cudaStreamSynchronize(s), res, exit);
  CUDACHECKGOTO(cudaStreamDestroy(s), res, exit);

  CUDACHECKGOTO(cudaMemcpy(&this->id, devBufDst, sizeof(uint64_t), cudaMemcpyDefault), res, exit);
  CUDACHECKGOTO(cudaFree(devBufSrc), res, exit);
  CUDACHECKGOTO(cudaFree(devBufDst), res, exit);

  *id = this->id++;

exit:
  this->m.unlock();
  return res;
}
