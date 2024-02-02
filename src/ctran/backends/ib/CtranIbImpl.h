// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_IMPL_H_
#define CTRAN_IB_IMPL_H_

#include <vector>
#include <thread>
#include <mutex>
#include <stdint.h>
#include <unordered_map>
#include <unordered_set>
#include "ibvwrap.h"
#include "bootstrap.h"

#define BOOTSTRAP_CMD_SETUP  (0)
#define BOOTSTRAP_CMD_TERMINATE  (1)

#define MAX_SEND_WR      (256)

struct ncclComm;

/**
 * Structure to describe a pending control operation
 */
struct PendingOp {
  enum PendingOpType {
    UNDEFINED,
    ISEND_CTRL,
    IRECV_CTRL,
  } type{UNDEFINED};
  struct {
    void* buf{nullptr};
    void* ibRegElem{nullptr};
    int peerRank{-1};
    CtranIbRequest* req{nullptr};
  } isendCtrl;
  struct {
    void** buf{nullptr};
    struct CtranIbRemoteAccessKey* key{nullptr};
    int peerRank{-1};
    CtranIbRequest* req{nullptr};
  } irecvCtrl;
};

/**
 * Singleton class to hold the IB network resources that are reused by all
 * communicators in the lifetime of program.
 */
class CtranIbSingleton {
  public:
    CtranIbSingleton(const CtranIbSingleton& obj) = delete;
    static CtranIbSingleton& getInstance();
    std::vector<int> ports;
    std::vector<struct ibv_context *> contexts;
    std::vector<struct ibv_pd *> pds;
    std::vector<std::string> devNames;

    // Record traffic in bytes whenever IB data transfer happens.
    // Accumulate per device name and per QP.
    void recordDeviceTraffic(struct ibv_context *ctx, size_t nbytes);
    void recordQpTraffic(struct ibv_qp* qp, size_t nbytes);

    // Record ncclComm reference to track missed comm destruction which may
    // cause failed CtranIbSingleton destruction (e.g., wrap_ibv_dealloc_pd
    // failed due to resource being used).
    void commRef(ncclComm* comm);
    void commDeref(ncclComm* comm);

    std::unordered_map<std::string, size_t> getDeviceTrafficSnapshot(void);
    std::unordered_map<uint32_t, size_t> getQpTrafficSnapshot(void);

  private:
    CtranIbSingleton();
    ~CtranIbSingleton();
    std::unordered_map<std::string, size_t> trafficPerDevice_;
    std::unordered_map<uint32_t, size_t> trafficPerQP_;
    std::mutex trafficRecordMutex_;

    std::unordered_set<ncclComm*> comms_;
    std::mutex commsMutex_;
};

class CtranIb::Impl {
public:
  Impl() = default;
  ~Impl() = default;

  static void bootstrapAccept(CtranIb::Impl *pimpl);
  ncclResult_t bootstrapConnect(int peerRank);
  ncclResult_t bootstrapTerminate();

  const char *ibv_wc_status_str(enum ibv_wc_status status);

  ncclComm* comm{nullptr};
  struct ibv_context* context{nullptr};
  struct ibv_pd* pd{nullptr};
  struct ibv_cq* cq{nullptr};
  int port{0};

  struct ncclSocket listenSocket;
  ncclSocketAddress *allListenSocketAddrs;
  std::thread listenThread;

  /* individual VCs for each peer */
  class VirtualConn;
  std::vector<class VirtualConn *> vcList;
  std::vector<uint32_t> numUnsignaledPuts;
  CtranIbRequest fakeReq;
  std::unordered_map<uint32_t, int> qpToRank;
  std::mutex m;

  std::vector<struct PendingOp *> pendingOps;

private:
  ncclResult_t bootstrapConnect(int peerRank, int cmd);
};

#endif
