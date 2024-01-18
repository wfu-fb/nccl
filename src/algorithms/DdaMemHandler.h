// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <unordered_map>
#include <vector>

#include "nccl.h"

namespace nccl {
namespace algorithms {

struct LocalMemAddr {
  void* addr{nullptr}; // local usable address
  bool isMmapped{false};
};

/**
 * DDA Memory Handler provides simple interface to allow
 * application to share memory address across ranks for
 * both multi-threaded/multi-process environment
 * sample usage:
 *   handler.add(void* addr1);
 *   handler.add(void* addr2);
 *
 *   handler.exchangeMemHandles();
 *
 *   auto addr = handler.get(remoteRank, addrIdx);
 */
class DdaMemHandler {
 public:
  DdaMemHandler(ncclComm_t comm);
  ~DdaMemHandler();

  // add local memory address, return index of the added address
  size_t add(void* addr);

  // exchange mem handles with other ranks in this communicator
  ncclResult_t exchangeMemHandles();

  // get address usable in local process that's either mmapped from
  // other process (multi-process) or return-as-is (multi-thread)
  void* get(int rank, int idx);

 private:
  // delete copy constructors
  DdaMemHandler(const DdaMemHandler&) = delete;
  DdaMemHandler& operator=(const DdaMemHandler&) = delete;

  ncclComm_t comm_{nullptr};
  std::unordered_map<int, std::vector<LocalMemAddr>> allMemAddrs_;
};

} // namespace algorithms
} // namespace nccl
