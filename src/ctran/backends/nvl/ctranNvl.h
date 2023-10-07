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
  ctranNvlRequest(void *addr, std::size_t len, ctranNvl *parent);
  ~ctranNvlRequest();

  void complete();
  ncclResult_t test(bool *isComplete);

  void *addr;
  std::size_t len;

private:
  class impl;
  std::unique_ptr<impl> pimpl;
};

class ctranNvl {
public:
  ctranNvl(ncclComm *comm);
  ~ctranNvl();
  ncclResult_t regMem(const void *buf, std::size_t len, void **hdl);
  ncclResult_t deregMem(const void *hdl);
  ncclResult_t isend(const void *buf, size_t len, int rank, const void *hdl, uint64_t commId, ctranNvlRequest **req);
  ncclResult_t irecv(void *buf, size_t len, int rank, const void *hdl, uint64_t commId, ctranNvlRequest **req);

protected:
  ncclResult_t progress(void);

private:
  class impl;
  std::unique_ptr<impl> pimpl;
  friend class ctranNvlRequest;
};

#endif
