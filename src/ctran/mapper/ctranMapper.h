// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_MAPPER_H_
#define CTRAN_MAPPER_H_

#include <memory>
#include <functional>
#include "nccl.h"
#include "checks.h"
#include "ctranIb.h"
#include "ctranNvl.h"

enum ctranBackend {
  CTRAN_BACKEND_UNSET,
  CTRAN_BACKEND_IB,
  CTRAN_BACKEND_NVL,
};

class ctranMapperMemPool {
 public:
  ctranMapperMemPool();
  ~ctranMapperMemPool();

  void printSnapshot();
  ncclResult_t init();
  ncclResult_t getBuf(std::size_t len, void** addr, void** hdl, std::size_t *bufLen);
  ncclResult_t getFreeBlk(std::size_t len, void** addr, void** hdl);
  ncclResult_t release(void* addr, void* hdl);
  ncclResult_t releaseAll();
  ncclResult_t alloc(void** addr, std::size_t len);
  ncclResult_t free(void* addr);
  ncclResult_t regMem(
      std::function<ncclResult_t(const void*, std::size_t, void**)> regMemFunc);
  ncclResult_t deregMem(std::function<ncclResult_t(void*)> deRegMemFunc);

 private:
  class impl;
  std::unique_ptr<class impl> pimpl;
};

class ctranMapperRequest {
public:
  ctranMapperRequest();
  ~ctranMapperRequest();
  ncclResult_t test(bool *isComplete);
  uint64_t getWaitTime();
  uint64_t getCommTime();

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
  ncclResult_t getTmpBuf(void** addr, std::size_t len, void **hdl);
  ncclResult_t releaseTmpBuf(void* addr, void *hdl);

  int rank;
  uint64_t commHash;
  uint64_t collId;

private:
  class impl;
  std::unique_ptr<impl> pimpl;
};

#endif
