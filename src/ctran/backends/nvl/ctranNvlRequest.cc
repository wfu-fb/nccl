// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranNvl.h"
#include "ctranNvlRequestImpl.h"

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
  this->pimpl->m.lock();
  this->pimpl->state = ctranNvlRequest::impl::COMPLETE;
  this->pimpl->m.unlock();
}

ncclResult_t ctranNvlRequest::test(bool *isComplete) {
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(this->pimpl->parent->progress(), res, exit);

  this->pimpl->m.lock();
  *isComplete = (this->pimpl->state == ctranNvlRequest::impl::COMPLETE);
  this->pimpl->m.unlock();

exit:
  return res;
}

