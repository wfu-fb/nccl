// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_H_
#define CTRAN_IB_H_

#include "nccl.h"
#include <memory>

class ctranIb;

enum ctranIbRequestTimestamp {
  REQ_POSTED,
  GOT_RTR,
  SEND_DATA_START,
  SEND_DATA_END,
};

class ctranIbRequest {
public:
  ctranIbRequest(void *addr, std::size_t len, void *hdl, ctranIb *parent);
  ~ctranIbRequest();

  void complete();
  ncclResult_t test(bool *isComplete);
  uint64_t getWaitTime();
  uint64_t getCommTime();

  void *addr;
  std::size_t len;
  void *hdl;

  void timestamp(ctranIbRequestTimestamp type);

private:
  class impl;
  std::unique_ptr<impl> pimpl;
  friend class ctranIb;
};

class ctranIb {
public:
  ctranIb(ncclComm *comm);
  ~ctranIb();

  ncclResult_t regMem(const void *buf, std::size_t len, void **hdl);
  ncclResult_t deregMem(void *hdl);
  ncclResult_t isend(const void *buf, size_t len, int rank, void *hdl, ctranIbRequest **req);
  ncclResult_t irecv(void *buf, size_t len, int rank, void *hdl, ctranIbRequest **req);

protected:
  ncclResult_t progress(void);

private:
  class impl;
  std::unique_ptr<impl> pimpl;
  friend class ctranIbRequest;
};

#endif
