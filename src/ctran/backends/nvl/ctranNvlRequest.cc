// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranNvl.h"

ctranNvlRequest::ctranNvlRequest(void *addr, std::size_t len) {
  this->state = ctranNvlRequest::INCOMPLETE;
}

ctranNvlRequest::~ctranNvlRequest() {
}

void ctranNvlRequest::complete() {
  this->state = ctranNvlRequest::COMPLETE;
}

ncclResult_t ctranNvlRequest::test(bool *isComplete) {
  *isComplete = (this->state == ctranNvlRequest::COMPLETE);

  return ncclSuccess;
}
