// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_REQUEST_IMPL_H_
#define CTRAN_IB_REQUEST_IMPL_H_

#include "ctranIb.h"
#include <chrono>

class ctranIbRequest::impl {
public:
  impl() = default;
  ~impl() = default;

  ctranIb *parent;

  enum {
    INCOMPLETE,
    COMPLETE,
  } state;

  std::chrono::time_point<std::chrono::high_resolution_clock> reqPosted;
  std::chrono::time_point<std::chrono::high_resolution_clock> gotRtr;
  std::chrono::time_point<std::chrono::high_resolution_clock> sendDataStart;
  std::chrono::time_point<std::chrono::high_resolution_clock> sendDataEnd;
  std::chrono::microseconds waitTime;
  std::chrono::microseconds commTime;
};

#endif
