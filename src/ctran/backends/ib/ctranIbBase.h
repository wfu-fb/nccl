// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_BASE_H_
#define CTRAN_IB_BASE_H_

#include <vector>
#include "ibvwrap.h"
#include "ctranIb.h"

struct controlMsg {
  uint64_t remoteAddr;
  uint32_t rkey;
};

struct controlWr {
  struct {
    struct {
      void *buf;
      void *hdl;
      ctranIbRequest *req;
    } send;
    struct {
      void **buf;
      struct ctranIbRemoteAccessKey *key;
      ctranIbRequest *req;
    } recv;
    struct {
      uint64_t remoteAddr;
      uint32_t rkey;
    } unex;
  } enqueued;
};

#endif
