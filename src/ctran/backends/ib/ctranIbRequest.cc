// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranIb.h"
#include "checks.h"
#include <iostream>

ctranIbRequest::ctranIbRequest() {
  this->state = ctranIbRequest::INCOMPLETE;
}

ctranIbRequest::~ctranIbRequest() {
}

void ctranIbRequest::complete() {
  this->state = ctranIbRequest::COMPLETE;
}

ncclResult_t ctranIbRequest::test(bool *isComplete) {
  *isComplete = (this->state == ctranIbRequest::COMPLETE);

  return ncclSuccess;
}
