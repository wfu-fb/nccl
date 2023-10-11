// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_NVL_IMPL_H_
#define CTRAN_NVL_IMPL_H_

#include <vector>
#include <unordered_map>
#include <thread>
#include <string.h>
#include <cuda_runtime.h>
#include "ctranNvl.h"

struct ctranNvlElem {
  enum elemType {
    ISEND,
    IRECV,
  } type;

  union {
    struct {
      const void *buf;
      std::size_t len;
    } isend;
    struct {
      void *buf;
      std::size_t len;
    } irecv;
  } u;

  ctranNvlRequest *req;
  cudaEvent_t e;
};

class ctranNvl::impl {
public:
  impl() = default;
  ~impl() = default;

  std::vector<struct ctranNvlElem *> postedSends;
  std::vector<struct ctranNvlElem *> postedRecvs;
  std::vector<struct ctranNvlElem *> pendingSends;
  std::vector<struct ctranNvlElem *> pendingRecvs;
  cudaStream_t s;
};

#endif
