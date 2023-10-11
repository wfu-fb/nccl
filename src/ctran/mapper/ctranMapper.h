// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_MAPPER_H_
#define CTRAN_MAPPER_H_

#include <mutex>
#include <memory>
#include <functional>
#include "nccl.h"
#include "checks.h"
#include "ctranIb.h"
#include "ctranNvl.h"

struct ctranMapperRemoteAccessKey {
  struct ctranIbRemoteAccessKey ibKey;
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
  std::unique_ptr<impl> pimpl;
};

class ctranMapper;
class ctranMapperRequest {
public:
  ctranMapperRequest(ctranMapper *mapper, cudaEvent_t e);
  ~ctranMapperRequest();
  ncclResult_t test(bool *isComplete);
  ncclResult_t wait();

  ctranIbRequest *ibReq;
  ctranNvlRequest *nvlReq;

private:
  cudaEvent_t e;
  ctranMapper *mapper;
  enum {
    INCOMPLETE,
    COMPLETE,
  } state;
};

struct ncclComm;

class ctranMapper {
public:
  ctranMapper(ncclComm *comm);
  ~ctranMapper();
  ncclResult_t regMem(const void *buf, std::size_t len, void **hdl);
  ncclResult_t deregMem(void *hdl);
  ncclResult_t searchRegHandle(const void *buf, std::size_t len, void **hdl);
  ncclResult_t icopy(void *dbuf, const void *sbuf, std::size_t len, ctranMapperRequest **req);
  ncclResult_t getTmpBuf(void** addr, std::size_t len, void **hdl);
  ncclResult_t releaseTmpBuf(void* addr, void *hdl);

  ncclResult_t isendCtrl(void *buf, void *hdl, int rank, ctranMapperRequest **req);
  ncclResult_t irecvCtrl(void **buf, struct ctranMapperRemoteAccessKey *key, int rank,
      ctranMapperRequest **req);
  ncclResult_t iput(const void *sbuf, void *dbuf, std::size_t len, int rank, void *shdl,
      struct ctranMapperRemoteAccessKey remoteAccessKey, bool notify, ctranMapperRequest **req);
  ncclResult_t checkNotify(int rank, bool *notify);
  ncclResult_t waitNotify(int rank);

  int rank;
  uint64_t commHash;

protected:
  ncclResult_t progress(void);

private:
  class impl;
  std::unique_ptr<impl> pimpl;
  std::mutex tmpBufLock;
  friend class ctranMapperRequest;
};

#endif
