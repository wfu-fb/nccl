// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranIb.h"
#include "ctranIbRequestImpl.h"
#include "checks.h"

ctranIbRequest::ctranIbRequest(void *addr, std::size_t len, void *hdl, ctranIb *parent) {
  this->pimpl = std::unique_ptr<impl>(new impl());

  this->addr = addr;
  this->len = len;
  this->hdl = hdl;

  this->pimpl->parent = parent;
  this->pimpl->state = ctranIbRequest::impl::INCOMPLETE;
}

ctranIbRequest::~ctranIbRequest() {
}

void ctranIbRequest::complete() {
  this->pimpl->m.lock();
  this->pimpl->state = ctranIbRequest::impl::COMPLETE;
  this->pimpl->m.unlock();
}

ncclResult_t ctranIbRequest::test(bool *isComplete) {
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(this->pimpl->parent->progress(), res, exit);

  this->pimpl->m.lock();
  *isComplete = (this->pimpl->state == ctranIbRequest::impl::COMPLETE);
  this->pimpl->m.unlock();

exit:
  return ncclSuccess;
}
