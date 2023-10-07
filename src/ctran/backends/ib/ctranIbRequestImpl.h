// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_REQUEST_IMPL_H_
#define CTRAN_IB_REQUEST_IMPL_H_

#include <mutex>
#include "ctranIb.h"

class ctranIbRequest::impl {
public:
  impl() = default;
  ~impl() = default;

  ctranIb *parent;

  enum {
    INCOMPLETE,
    COMPLETE,
  } state;
  std::mutex m;
};

#endif
