// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_BASE_H_
#define CTRAN_IB_BASE_H_

#include <vector>
#include "ibvwrap.h"
#include "ctranIb.h"

// FIXME: RoCE verbs communication seems to fail when sending small
// messages (up to 32 bytes).  My best guess is that there is a
// setting that is causing it to receive data "inline" with the WQE,
// which the poll_cq is trying to copy into the final buffer.  If the
// final buffer is on the GPU, this is causing a segmentation fault.
// We are working around that by padding the control message to be at
// least 33 bytes.  Unfortunately, this number could be device and
// driver specific, so it is risky to rely on this hardcoded value. --
// Pavan Balaji (9/5/2023)
#define MIN_CONTROL_MSG_SIZE (33)
struct controlMsg {
  union {
    struct {
      uint64_t remoteAddr;
      uint32_t rkey;
    } msg;
    char padding[MIN_CONTROL_MSG_SIZE];
  } u;
};

struct wqeState {
  enum wqeType {
    CONTROL_SEND,
    CONTROL_RECV,
    DATA_SEND,
    DATA_RECV,
  } wqeType;
  int peerRank;

  /* the sendControl message and recvData message form a pair
   * (receiving data requires sending a control message) and they both
   * need to be processed in lock-step.  Similarly the recvControl
   * message and the sendData message form a pair (sending data
   * requires receiving a control message). */
  struct wqeState *peerWqe;

  union {
    struct {
      struct controlMsg *cmsg;
    } control;
    struct {
      uint64_t remoteAddr;
      uint32_t rkey;
      ctranIbRequest *req;
    } data;
  } u;

  uint64_t wqeId;
};

static const char *wqeName(enum wqeState::wqeType wqeType) {
  switch (wqeType) {
    case wqeState::wqeType::CONTROL_SEND: return "CONTROL_SEND";
    case wqeState::wqeType::CONTROL_RECV: return "CONTROL_RECV";
    case wqeState::wqeType::DATA_SEND: return "DATA_SEND";
    case wqeState::wqeType::DATA_RECV: return "DATA_RECV";
    default: return "UNKNOWN";
  }
}

#endif
