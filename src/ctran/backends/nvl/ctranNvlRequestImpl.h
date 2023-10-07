// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_NVL_REQUEST_IMPL_H_
#define CTRAN_NVL_REQUEST_IMPL_H_

#include <mutex>
#include "ctranNvl.h"

class ctranNvlRequest::impl {
public:
  impl() = default;
  ~impl() = default;

  ctranNvl *parent;
  enum {
    INCOMPLETE,
    COMPLETE,
  } state;
  std::mutex m;
};

#endif
