// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include "CtranIb.h"
#include "checks.h"

void CtranIbRequest::setRefCount(int refCount) {
  this->refCount_ = refCount;
}

void CtranIbRequest::complete() {
  this->refCount_--;
  if (this->refCount_ == 0) {
    this->state_ = CtranIbRequest::COMPLETE;
  }
}

bool CtranIbRequest::isComplete() {
  return this->state_ == CtranIbRequest::COMPLETE;
}
