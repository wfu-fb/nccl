// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_MEMPOOL_IMPL_H_
#define CTRAN_MEMPOOL_IMPL_H_

#include <cstdint>
#include <memory>
#include <unordered_set>
#include "ctranMapper.h"
#include "ctranAvlTree.h"

constexpr std::size_t kDefaultBlockSize = 1024 * 1024; // 2MB
constexpr std::size_t kDefaultMinBlockSize = 4096; // 4KB
constexpr int kDefaultNumBlocks = 8;

// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
class memBlockInfo {
 public:
  memBlockInfo(void* _addr, std::size_t _len, int _devIdx, void* _hdl)
      : addr(_addr), len(_len), devIdx(_devIdx), hdl(_hdl) {}
  void* addr;
  std::size_t len;
  int devIdx;
  void* hdl;
};

class ctranMapperMemPool::impl {
 public:
  impl() = default;
  ~impl() = default;

  // TODO: use small and large pools or segements of large allocation? to reduce
  // the waste
  std::unordered_set<std::shared_ptr<memBlockInfo>> freePool;
  std::unordered_set<std::shared_ptr<memBlockInfo>> busyPool;
  size_t blockSize{kDefaultBlockSize};
  int numBlocks{kDefaultNumBlocks};
};

#endif
