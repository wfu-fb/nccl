// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_MAPPER_IMPL_H_
#define CTRAN_MAPPER_IMPL_H_

#include <mutex>
#include "ctranIb.h"
#include "ctranNvl.h"
#include "ctranMapper.h"
#include "ctranRegCache.h"

class ctranMapperShared {
  public:
    // This should not be public, but the compiler does not seem to
    // like creating a shared pointer to this object if it's a private
    // constructor
    ctranMapperShared();

  private:
    ncclResult_t getUniqueId(ncclComm *comm, uint64_t *id);
    uint64_t id;
    std::mutex m;
    friend class ctranMapper;
};

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

  std::shared_ptr<class ctranMapperShared> sharedMapper;
  uint64_t uniqueId;
  std::vector<int> rankMap;
  std::vector<enum ctranBackend> rankBackendMap;

  class ctranRegCache *regCache;

  std::vector<enum ctranMapperBackend> backends;
  std::shared_ptr<class ctranIb> ctranIb;
  std::shared_ptr<class ctranNvl> ctranNvl;

  cudaStream_t s;
};

#endif
