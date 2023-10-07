// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "checks.h"
#include "ctranMapper.h"
#include "ctranMapperMemPoolImpl.h"
#include "debug.h"

ctranMapperMemPool::ctranMapperMemPool() {
  this->pimpl = std::unique_ptr<ctranMapperMemPool::impl>(new ctranMapperMemPool::impl());
  this->init();
}

ctranMapperMemPool::~ctranMapperMemPool() = default;

void ctranMapperMemPool::printSnapshot() {
  INFO(
      NCCL_INIT,
      "CTRAN-MEM: numBlocks %d, freePool has %lu blocks, busyPool has %lu blocks",
      this->pimpl->numBlocks,
      this->pimpl->freePool.size(),
      this->pimpl->busyPool.size());
}

ncclResult_t ctranMapperMemPool::init() {
  // create a pool
  this->pimpl->freePool.reserve(this->pimpl->numBlocks);
  this->pimpl->busyPool.reserve(this->pimpl->numBlocks);
  for (int i = 0; i < this->pimpl->numBlocks; ++i) {
    void* addr = nullptr;
    this->alloc(&addr, this->pimpl->blockSize);
    // TODO: do we need devId and handl?
    this->pimpl->freePool.emplace(
        std::make_shared<memBlockInfo>(addr, this->pimpl->blockSize, -1, nullptr));
  }
  INFO(
      NCCL_INIT,
      "CTRAN: initialized memory pool with %d blocks",
      this->pimpl->numBlocks);

  this->printSnapshot();
  return ncclSuccess;
}

ncclResult_t ctranMapperMemPool::releaseAll() {
  if (!this->pimpl->freePool.empty() || !this->pimpl->busyPool.empty()) {
    INFO(NCCL_INIT, "free memory pool!");
    this->printSnapshot();
    if (!this->pimpl->busyPool.empty()) {
      WARN(
          "[WARN] %s:%d: %lu blocks are not released",
          __FILE__,
          __LINE__,
          this->pimpl->busyPool.size());
    }
    for (auto& blk : this->pimpl->busyPool) {
      this->free(blk->addr);
    }
    for (auto& blk : this->pimpl->freePool) {
      this->free(blk->addr);
    }
    this->pimpl->busyPool.clear();
    this->pimpl->freePool.clear();
    this->pimpl->numBlocks = 0;
  }
  return ncclSuccess;
}

/* Get a large enough block from free list
 * If there is no free block large enough, allocate a new one.
 * TODO:
 *  1. use a better algorithm to find a free block
 *  2. use segment to reduce waste of memory if user requests small buffer
 *  3. perform registration along with allocation?
 *  4. ensure thread safe
 */
ncclResult_t
ctranMapperMemPool::getFreeBlk(std::size_t len, void** addr, void** hdl) {
  *hdl = nullptr;
  // search the free list to find a block that is large enough
  // TODO: more efficient search
  bool found = false;
  for (auto& blk : this->pimpl->freePool) {
    if (blk->len >= len) {
      TRACE(
          "CTRAN-MEM: found a free block %p (hdl %p), %lu >= size %lu",
          blk->addr,
          blk->hdl,
          blk->len,
          len);
      *addr = blk->addr;
      if (blk->hdl != nullptr) {
        *hdl = blk->hdl;
      }
      this->pimpl->busyPool.emplace(std::move(blk));
      this->pimpl->freePool.erase(blk);
      found = true;
      break;
    }
  }
  if (!found) {
    auto newLen = (len < kDefaultMinBlockSize) ? kDefaultMinBlockSize : len;
    NCCLCHECK(this->alloc(addr, newLen));
    this->pimpl->busyPool.emplace(std::make_shared<memBlockInfo>(*addr, newLen, -1, nullptr));
    TRACE(
        "%s:%d: no free block available to satisify %lu bytes buffer, allocate a new one %p",
        __FILE__,
        __LINE__,
        newLen,
        *addr);
    ++this->pimpl->numBlocks;
    this->printSnapshot();
  }

  return ncclSuccess;
}

ncclResult_t ctranMapperMemPool::getBuf(std::size_t len, void** addr, void** hdl, std::size_t *bufLen) {
  this->getFreeBlk(len, addr, hdl);
  *bufLen = (len < kDefaultMinBlockSize) ? kDefaultMinBlockSize : len;
  return ncclSuccess;
}

ncclResult_t ctranMapperMemPool::release(void* addr, void* hdl) {
  // TODO: more efficient search
  for (auto& blk : this->pimpl->busyPool) {
    if (blk->addr == addr) {
      TRACE(
          "CTRAN-MEM: release a block %p (hdl %p), size %lu, back to free pool",
          blk->addr,
          blk->hdl,
          blk->len);
      if (!blk->hdl && hdl) {
        blk->hdl = hdl;
      }
      this->pimpl->freePool.emplace(std::move(blk));
      this->pimpl->busyPool.erase(blk);
      break;
    }
  }

  return ncclSuccess;
}

ncclResult_t ctranMapperMemPool::alloc(void** addr, std::size_t len) {
  CUDACHECK(cudaMalloc(addr, len));
  return ncclSuccess;
}

ncclResult_t ctranMapperMemPool::free(void* addr) {
  CUDACHECK(cudaFree(addr));
  return ncclSuccess;
}

ncclResult_t ctranMapperMemPool::regMem(
    std::function<ncclResult_t(const void*, std::size_t, void**)> regMemFunc) {
  for (auto& blk : this->pimpl->freePool) {
    if (!blk->hdl) {
      void* hdl;
      NCCLCHECK(regMemFunc(blk->addr, blk->len, &hdl));
      blk->hdl = hdl;
    }
  }
  for (auto& blk : this->pimpl->busyPool) {
    if (!blk->hdl) {
      void* hdl;
      NCCLCHECK(regMemFunc(blk->addr, blk->len, &hdl));
      blk->hdl = hdl;
    }
  }
  return ncclSuccess;
}

ncclResult_t ctranMapperMemPool::deregMem(
    std::function<ncclResult_t(void*)> deRegMemFunc) {
  for (auto& blk : this->pimpl->freePool) {
    if (blk->hdl) {
      TRACE("deregister %p, hdl %p", blk->addr, blk->hdl);
      NCCLCHECK(deRegMemFunc(blk->hdl));
    }
  }
  for (auto& blk : this->pimpl->busyPool) {
    if (blk->hdl) {
      TRACE("deregister %p, hdl %p", blk->addr, blk->hdl);
      NCCLCHECK(deRegMemFunc(blk->hdl));
    }
  }
  return ncclSuccess;
}
