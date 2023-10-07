// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_NVL_IMPL_H_
#define CTRAN_NVL_IMPL_H_

#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <string.h>
#include <cuda_runtime.h>
#include "ctranNvl.h"

struct ctranNvlElem {
  enum elemType {
    ISEND,
    IRECV,
  } type;
  uint64_t commId;

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

struct ctranNvlCommQueues {
  uint64_t commId;
  std::vector<struct ctranNvlElem *> postedSends;
  std::vector<struct ctranNvlElem *> postedRecvs;
  std::vector<struct ctranNvlElem *> pendingSends;
  std::vector<struct ctranNvlElem *> pendingRecvs;
};

class ctranNvl::impl {
public:
  impl() = default;
  ~impl() = default;

  std::unordered_map<uint64_t, ctranNvlCommQueues *> ops;
  cudaStream_t s;
  std::mutex m;
};

#endif
