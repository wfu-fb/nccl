// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_MAPPER_H_
#define CTRAN_MAPPER_H_

#include <memory>
#include "nccl.h"
#include "checks.h"
#include "ctranIb.h"
#include "ctranNvl.h"

enum ctranBackend {
  CTRAN_BACKEND_UNSET,
  CTRAN_BACKEND_IB,
  CTRAN_BACKEND_NVL,
};

class ctranMapperRequest {
public:
  ctranMapperRequest();
  ~ctranMapperRequest();
  ncclResult_t test(bool *isComplete);

  ctranIbRequest *ibReq;
  ctranNvlRequest *nvlReq;
  cudaEvent_t e;

private:
  class impl;
  std::unique_ptr<impl> pimpl;
};

struct ncclComm;

class ctranMapper {
public:
  ctranMapper(ncclComm *comm, ncclComm *parent, int *parentRanks);
  ~ctranMapper();
  ncclResult_t regMem(const void *buf, std::size_t len, void **hdl);
  ncclResult_t deregMem(void *hdl);
  ncclResult_t searchRegHandle(const void *buf, std::size_t len, void **hdl);
  ncclResult_t isend(const void *buf, std::size_t len, int rank, void *hdl, ctranMapperRequest **req);
  ncclResult_t irecv(void *buf, std::size_t len, int rank, void *hdl, ctranMapperRequest **req);
  ncclResult_t icopy(void *dbuf, const void *sbuf, std::size_t len, ctranMapperRequest **req);

private:
  class impl;
  std::unique_ptr<impl> pimpl;
};

#endif
