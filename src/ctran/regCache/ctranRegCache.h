// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_REG_CACHE_H_
#define CTRAN_REG_CACHE_H_

#include <cstdint>
#include <vector>
#include <memory>
#include "nccl.h"

class ctranRegCache {
public:
  ctranRegCache();
  ~ctranRegCache();

  ncclResult_t insert(const void *addr, std::size_t len, void *val, void **hdl);
  ncclResult_t remove(void *hdl);
  ncclResult_t search(const void *addr, std::size_t len, void **hdl);
  ncclResult_t lookup(void *hdl, void **val);
  std::vector<void *> flush();

private:
  class impl;
  std::unique_ptr<class impl> pimpl;
};

#endif
