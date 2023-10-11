// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_MAPPER_IMPL_H_
#define CTRAN_MAPPER_IMPL_H_

#include "ctranIb.h"
#include "ctranNvl.h"
#include "ctranMapper.h"
#include "ctranRegCache.h"

struct ctranMapperRegElem {
  void *ibHdl;
  void *nvlHdl;
};

enum ctranMapperBackend {
  NVL,
  IB,
};

class ctranMapper::impl {
public:
  impl() = default;
  ~impl() = default;

  std::vector<enum ctranBackend> rankBackendMap;

  class ctranRegCache *regCache;
  class ctranMapperMemPool *memPool;

  std::vector<enum ctranMapperBackend> backends;
  std::unique_ptr<class ctranIb> ctranIb;
  std::unique_ptr<class ctranNvl> ctranNvl;

  cudaStream_t s;
};

#endif
