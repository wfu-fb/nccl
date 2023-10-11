// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_IMPL_H_
#define CTRAN_IB_IMPL_H_

#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include "ibvwrap.h"
#include "bootstrap.h"

#define BOOTSTRAP_CMD_SETUP  (0)
#define BOOTSTRAP_CMD_TERMINATE  (1)

class ctranIb::impl {
public:
  impl() = default;
  ~impl() = default;

  static void bootstrapAccept(ctranIb::impl *pimpl);
  ncclResult_t bootstrapConnect(int peerRank);
  ncclResult_t bootstrapTerminate();

  const char *ibv_wc_status_str(enum ibv_wc_status status);

  int rank;
  int nRanks;
  struct ibv_context *context;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  int port;

  struct ncclSocket listenSocket;
  ncclSocketAddress *allListenSocketAddrs;
  std::thread listenThread;

  /* individual VCs for each peer */
  class vc;
  std::vector<class vc *> vcList;

private:
  ncclResult_t bootstrapConnect(int peerRank, int cmd);
  std::mutex bootstrapMutex;
};

#endif
