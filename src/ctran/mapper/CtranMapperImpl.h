// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_MAPPER_IMPL_H_
#define CTRAN_MAPPER_IMPL_H_

#include "CtranAvlTree.h"
#include "CtranIb.h"
#include "CtranMapper.h"

enum CtranMapperRegElemState {
  CACHED,
  REGISTERED,
};

struct CtranMapperRegElem {
  const void* buf;
  std::size_t len;
  void* ibRegElem;
  enum CtranMapperRegElemState state;
};

enum CtranMapperBackend {
  UNSET,
  IB,
};

class CtranMapper::impl {
 public:
  impl() = default;
  ~impl() = default;

  ncclResult_t regMem(struct CtranMapperRegElem* mapperRegElem);
  ncclResult_t deregMem(struct CtranMapperRegElem* mapperRegElem);

  std::unique_ptr<class CtranAvlTree> mapperRegElemList;

  std::vector<enum CtranMapperBackend> rankBackendMap;
  std::vector<enum CtranMapperBackend> backends;
  std::unique_ptr<class CtranIb> ctranIb;
};

#endif
