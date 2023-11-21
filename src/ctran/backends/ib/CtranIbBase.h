// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_BASE_H_
#define CTRAN_IB_BASE_H_

#include <vector>
#include "ibvwrap.h"
#include "CtranIb.h"

/**
 * Structure of control message transferred by isendCtrl/irecvCtrl.
 */
struct ControlMsg {
  uint64_t remoteAddr{0};
  uint32_t rkey{0};
};

/**
 * Structure of the work request (WR) describing a pending control message.
 */
struct ControlWr {
  struct {
    struct {
      void* buf{nullptr};
      void* ibRegElem{nullptr};
      CtranIbRequest* req{nullptr};
    } send;
    struct {
      void** buf{nullptr};
      struct CtranIbRemoteAccessKey* key{nullptr};
      CtranIbRequest* req{nullptr};
    } recv;
    struct {
      uint64_t remoteAddr{0};
      uint32_t rkey{0};
    } unex;
  } enqueued;
};

#endif
