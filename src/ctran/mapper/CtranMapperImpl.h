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

  uint32_t numRegistrations; /* number of currently registered buffers */
  uint32_t numCachedRegistrations; /* number of currently cached but not yet registered buffers in lazy registration; buffer still pre-registered by user. */
  uint32_t totalNumDynamicRegistrations; /* total number of buffers at lifetime that were not pre-registered by user but temporarily in communication */
  uint32_t totalNumRegistrations; /* total number of registered buffers at lifetime */
  uint32_t totalNumCachedRegistrations; /* total number of cached buffers at lifetime */
  uint32_t totalNumRegLookupHit; /* total number of lookup calls to search buffer registration and found registered handle */
  uint32_t totalNumRegLookupMiss; /* total number of lookup calls to search buffer registration and could
                                   * not found registered handle (i.e., by either lazy registration or dynamic registration )*/
};

#endif
