// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_REG_CACHE_IMPL_H_
#define CTRAN_REG_CACHE_IMPL_H_

#include <cstdint>
#include <vector>
#include "ctranRegCache.h"

struct regElem {
  uintptr_t addr;
  std::size_t len;
  void *val;
};

class ctranRegCache::impl {
public:
  impl() = default;
  ~impl() = default;

  std::vector<regElem *> root;
};

#endif
