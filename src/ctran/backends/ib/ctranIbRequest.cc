// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranIb.h"
#include "checks.h"
#include <iostream>

ctranIbRequest::ctranIbRequest() {
  this->state = ctranIbRequest::INCOMPLETE;
  this->refCount = 1;
}

ctranIbRequest::~ctranIbRequest() {
}

void ctranIbRequest::setRefCount(int refCount) {
  this->refCount = refCount;
}

void ctranIbRequest::complete() {
  this->refCount--;
  if (this->refCount == 0) {
    this->state = ctranIbRequest::COMPLETE;
  }
}

ncclResult_t ctranIbRequest::test(bool *isComplete) {
  ncclResult_t res = ncclSuccess;

  *isComplete = (this->state == ctranIbRequest::COMPLETE);

  return res;
}
