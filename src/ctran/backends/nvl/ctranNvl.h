// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_NVL_H_
#define CTRAN_NVL_H_

#include <memory>
#include "ctranNvl.h"
#include "nccl.h"
#include "checks.h"

class ctranNvl;
struct ncclComm;

class ctranNvlRequest {
public:
  ctranNvlRequest(void *addr, std::size_t len);
  ~ctranNvlRequest();

  void complete();
  ncclResult_t test(bool *isComplete);

private:
  enum {
    INCOMPLETE,
    COMPLETE,
  } state;
};

class ctranNvl {
public:
  ctranNvl(ncclComm *comm);
  ~ctranNvl();
  ncclResult_t regMem(const void *buf, std::size_t len, void **nvlRegElem);
  ncclResult_t deregMem(const void *nvlRegElem);
  ncclResult_t isend(const void *buf, size_t len, int rank, const void *nvlRegElem, ctranNvlRequest **req);
  ncclResult_t irecv(void *buf, size_t len, int rank, const void *nvlRegElem, ctranNvlRequest **req);
  ncclResult_t progress(void);

private:
  class impl;
  std::unique_ptr<impl> pimpl;
};

#endif
