// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include <vector>
#include <thread>
#include <unistd.h>
#include "nccl.h"
#include "checks.h"
#include "ctranIb.h"
#include "ctranIbImpl.h"
#include "ctranIbVc.h"

void ctranIb::impl::bootstrapAccept(ctranIb::impl *pimpl) {
  while (1) {
    struct ncclSocket sock;
    int peerRank;
    int cmd;

    NCCLCHECKIGNORE(ncclSocketInit(&sock));
    NCCLCHECKIGNORE(ncclSocketAccept(&sock, &pimpl->listenSocket));
    NCCLCHECKIGNORE(ncclSocketRecv(&sock, &cmd, sizeof(int)));
    NCCLCHECKIGNORE(ncclSocketRecv(&sock, &peerRank, sizeof(int)));

    if (cmd == BOOTSTRAP_CMD_TERMINATE) {
      NCCLCHECKIGNORE(ncclSocketClose(&sock));
      break;
    }

    auto vc = pimpl->vcList[peerRank];

    pimpl->bootstrapMutex.lock();

    /* exchange business cards */
    std::size_t size;
    void *localBusCard, *remoteBusCard;
    size = vc->getBusCardSize();
    localBusCard = malloc(size);
    remoteBusCard = malloc(size);
    NCCLCHECKIGNORE(vc->getLocalBusCard(localBusCard));
    NCCLCHECKIGNORE(ncclSocketRecv(&sock, remoteBusCard, size));
    NCCLCHECKIGNORE(ncclSocketSend(&sock, localBusCard, size));
    NCCLCHECKIGNORE(vc->setupVc(remoteBusCard));
    free(localBusCard);
    free(remoteBusCard);

    /* Ack that the connection is fully established */
    int ack;
    NCCLCHECKIGNORE(ncclSocketSend(&sock, &ack, sizeof(int)));
    NCCLCHECKIGNORE(ncclSocketRecv(&sock, &ack, sizeof(int)));

    NCCLCHECKIGNORE(ncclSocketClose(&sock));

    pimpl->bootstrapMutex.unlock();
  }
}

ncclResult_t ctranIb::impl::bootstrapConnect(int peerRank, int cmd) {
  ncclResult_t res = ncclSuccess;
  auto vc = this->vcList[peerRank];

  this->bootstrapMutex.lock();

  struct ncclSocket sock;
  NCCLCHECKGOTO(ncclSocketInit(&sock, &allListenSocketAddrs[peerRank]), res, exit);
  NCCLCHECKGOTO(ncclSocketConnect(&sock), res, exit);
  NCCLCHECKGOTO(ncclSocketSend(&sock, &cmd, sizeof(int)), res, exit);
  NCCLCHECKGOTO(ncclSocketSend(&sock, &this->rank, sizeof(int)), res, exit);

  if (peerRank == this->rank) {
    NCCLCHECKGOTO(ncclSocketClose(&sock), res, exit);
    goto exit;
  }

  /* exchange business cards */
  std::size_t size;
  void *localBusCard, *remoteBusCard;
  size = vc->getBusCardSize();
  localBusCard = malloc(size);
  remoteBusCard = malloc(size);
  NCCLCHECKGOTO(vc->getLocalBusCard(localBusCard), res, exit);
  NCCLCHECKGOTO(ncclSocketSend(&sock, localBusCard, size), res, exit);
  NCCLCHECKGOTO(ncclSocketRecv(&sock, remoteBusCard, size), res, exit);
  NCCLCHECKGOTO(vc->setupVc(remoteBusCard), res, exit);
  free(localBusCard);
  free(remoteBusCard);

  /* Ack that the connection is fully established */
  int ack;
  NCCLCHECKGOTO(ncclSocketRecv(&sock, &ack, sizeof(int)), res, exit);
  NCCLCHECKGOTO(ncclSocketSend(&sock, &ack, sizeof(int)), res, exit);

  NCCLCHECKGOTO(ncclSocketClose(&sock), res, exit);

exit:
  this->bootstrapMutex.unlock();
  return res;
}

ncclResult_t ctranIb::impl::bootstrapConnect(int peerRank) {
  return this->bootstrapConnect(peerRank, BOOTSTRAP_CMD_SETUP);
}


ncclResult_t ctranIb::impl::bootstrapTerminate() {
  return this->bootstrapConnect(this->rank, BOOTSTRAP_CMD_TERMINATE);
}

const char *ctranIb::impl::ibv_wc_status_str(enum ibv_wc_status status) {
  switch (status) {
    case IBV_WC_SUCCESS: return "success";
    case IBV_WC_LOC_LEN_ERR: return "local length error";
    case IBV_WC_LOC_QP_OP_ERR: return "local QP operation error";
    case IBV_WC_LOC_EEC_OP_ERR: return "local EE context operation error";
    case IBV_WC_LOC_PROT_ERR: return "local protection error";
    case IBV_WC_WR_FLUSH_ERR: return "Work Request Flushed Error";
    case IBV_WC_MW_BIND_ERR: return "memory management operation error";
    case IBV_WC_BAD_RESP_ERR: return "bad response error";
    case IBV_WC_LOC_ACCESS_ERR: return "local access error";
    case IBV_WC_REM_INV_REQ_ERR: return "remote invalid request error";
    case IBV_WC_REM_ACCESS_ERR: return "remote access error";
    case IBV_WC_REM_OP_ERR: return "remote operation error";
    case IBV_WC_RETRY_EXC_ERR: return "transport retry counter exceeded";
    case IBV_WC_RNR_RETRY_EXC_ERR: return "RNR retry counter exceeded";
    case IBV_WC_LOC_RDD_VIOL_ERR: return "local RDD violation error";
    case IBV_WC_REM_INV_RD_REQ_ERR: return "remote invalid RD request";
    case IBV_WC_REM_ABORT_ERR: return "aborted error";
    case IBV_WC_INV_EECN_ERR: return "invalid EE context number";
    case IBV_WC_INV_EEC_STATE_ERR: return "invalid EE context state";
    case IBV_WC_FATAL_ERR: return "fatal error";
    case IBV_WC_RESP_TIMEOUT_ERR: return "response timeout error";
    case IBV_WC_GENERAL_ERR: return "general error";
    default: return "unrecognized error";
  }
}
