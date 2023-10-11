// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranNvl.h"
#include "ctranNvlRequestImpl.h"
#include <chrono>

ctranNvlRequest::ctranNvlRequest(void *addr, std::size_t len, ctranNvl *parent) {
  this->pimpl = std::unique_ptr<impl>(new impl());

  this->addr = addr;
  this->len = len;

  this->pimpl->parent = parent;
  this->pimpl->state = ctranNvlRequest::impl::INCOMPLETE;
}

ctranNvlRequest::~ctranNvlRequest() {
}

void ctranNvlRequest::complete() {
  this->pimpl->state = ctranNvlRequest::impl::COMPLETE;
}

ncclResult_t ctranNvlRequest::test(bool *isComplete) {
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(this->pimpl->parent->progress(), res, exit);

  *isComplete = (this->pimpl->state == ctranNvlRequest::impl::COMPLETE);

exit:
  return res;
}

uint64_t ctranNvlRequest::getWaitTime() {
  return std::chrono::microseconds::zero().count();
}

uint64_t ctranNvlRequest::getCommTime() {
  return std::chrono::microseconds::zero().count();
}
