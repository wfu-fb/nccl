// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_MAPPER_IMPL_H_
#define CTRAN_MAPPER_IMPL_H_

#include "ctranIb.h"
#include "ctranNvl.h"
#include "ctranMapper.h"
#include "ctranAvlTree.h"

struct ctranMapperRegElem {
  void *ibRegElem;
  void *nvlRegElem;
};

enum ctranMapperBackend {
  UNSET,
  NVL,
  IB,
};

class ctranMapper::impl {
public:
  impl() = default;
  ~impl() = default;

  class ctranAvlTree *mapperRegElemList;
  class ctranMapperMemPool *memPool;

  std::vector<enum ctranMapperBackend> rankBackendMap;
  std::vector<enum ctranMapperBackend> backends;
  std::unique_ptr<class ctranIb> ctranIb;
  std::unique_ptr<class ctranNvl> ctranNvl;
};

#endif
