// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "DdaThreadedData.h"

namespace nccl {
namespace algorithms {

// intialize static members
DdaThreadedData* DdaThreadedData::data_ = nullptr;
std::mutex DdaThreadedData::instanceMutex_;
std::unordered_map<uint64_t, std::unordered_set<int>>
    DdaThreadedData::commToRanks_ = {};

DdaThreadedData::DdaThreadedData() {}

DdaThreadedData::~DdaThreadedData() {
  std::lock_guard<std::mutex> lock(instanceMutex_);
  if (data_ != nullptr) {
    delete data_;
  }
}

DdaThreadedData* DdaThreadedData::get() {
  std::lock_guard<std::mutex> lock(instanceMutex_);
  if (data_ != nullptr) {
    return data_;
  }
  data_ = new DdaThreadedData();
  return data_;
}

void DdaThreadedData::clear() {
  std::lock_guard<std::mutex> lock(instanceMutex_);
  commToRanks_.clear();
}

void DdaThreadedData::clear(uint64_t commHash) {
  std::lock_guard<std::mutex> lock(instanceMutex_);
  auto registeredRanks = commToRanks_.find(commHash);
  if (registeredRanks == commToRanks_.end()) {
    return;
  }
  registeredRanks->second.clear();
}

bool DdaThreadedData::registerRank(uint64_t commHash, int rank) {
  std::lock_guard<std::mutex> lock(instanceMutex_);
  bool inserted = false;
  std::tie(std::ignore, inserted) = commToRanks_[commHash].insert(rank);
  return inserted;
}

bool DdaThreadedData::unregisterRank(uint64_t commHash, int rank) {
  std::lock_guard<std::mutex> lock(instanceMutex_);
  auto registeredRanks = commToRanks_.find(commHash);
  if (registeredRanks == commToRanks_.end()) {
    return false;
  }
  return registeredRanks->second.erase(rank) > 0;
}

bool DdaThreadedData::hasRank(uint64_t commHash, int rank) {
  std::lock_guard<std::mutex> lock(instanceMutex_);
  auto registeredRanks = commToRanks_.find(commHash);
  if (registeredRanks == commToRanks_.end()) {
    return false;
  }
  return registeredRanks->second.count(rank) > 0;
}

size_t DdaThreadedData::numRanks(uint64_t commHash) {
  std::lock_guard<std::mutex> lock(instanceMutex_);
  auto registeredRanks = commToRanks_.find(commHash);
  if (registeredRanks == commToRanks_.end()) {
    return 0;
  }
  return registeredRanks->second.size();
}

} // namespace algorithms
} // namespace nccl
