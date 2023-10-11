// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranIb.h"
#include "ctranIbRequestImpl.h"
#include "checks.h"
#include <iostream>

ctranIbRequest::ctranIbRequest(void *addr, std::size_t len, void *hdl, ctranIb *parent) {
  this->pimpl = std::unique_ptr<impl>(new impl());

  this->addr = addr;
  this->len = len;
  this->hdl = hdl;

  this->pimpl->parent = parent;
  this->pimpl->state = ctranIbRequest::impl::INCOMPLETE;

  this->pimpl->waitTime = std::chrono::microseconds::zero();
  this->pimpl->commTime = std::chrono::microseconds::zero();
}

ctranIbRequest::~ctranIbRequest() {
}

void ctranIbRequest::complete() {
  this->pimpl->state = ctranIbRequest::impl::COMPLETE;
}

void ctranIbRequest::timestamp(ctranIbRequestTimestamp type) {
  if (type == ctranIbRequestTimestamp::REQ_POSTED) {
    this->pimpl->reqPosted = std::chrono::high_resolution_clock::now();
  } else if (type == ctranIbRequestTimestamp::GOT_RTR) {
    this->pimpl->gotRtr = std::chrono::high_resolution_clock::now();
  } else if (type == ctranIbRequestTimestamp::SEND_DATA_START) {
    this->pimpl->sendDataStart = std::chrono::high_resolution_clock::now();
  } else if (type == ctranIbRequestTimestamp::SEND_DATA_END) {
    this->pimpl->sendDataEnd = std::chrono::high_resolution_clock::now();
  }
}

ncclResult_t ctranIbRequest::test(bool *isComplete) {
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(this->pimpl->parent->progress(), res, exit);

  *isComplete = (this->pimpl->state == ctranIbRequest::impl::COMPLETE);

  if (*isComplete == true) {
    this->pimpl->waitTime = std::chrono::duration_cast<std::chrono::microseconds>
            (this->pimpl->gotRtr - this->pimpl->reqPosted);
    this->pimpl->commTime = std::chrono::duration_cast<std::chrono::microseconds>
      (this->pimpl->sendDataEnd - this->pimpl->sendDataStart);
  }

exit:
  return ncclSuccess;
}

uint64_t ctranIbRequest::getWaitTime() {
  return static_cast<uint64_t>(this->pimpl->waitTime.count());
}

uint64_t ctranIbRequest::getCommTime() {
  return static_cast<uint64_t>(this->pimpl->commTime.count());
}
