// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_VC_H_
#define CTRAN_IB_VC_H_

#include <vector>
#include <unordered_map>
#include <mutex>
#include <deque>
#include "ibvwrap.h"
#include "ctranIbBase.h"

#define MAX_CONTROL_MSGS (128)

class ctranIb::impl::vc {
  public:
    vc(struct ibv_context *context, struct ibv_pd *pd, struct ibv_cq *cq,
        int port, int peerRank);
    ~vc();

    bool isReady();
    std::size_t getBusCardSize();
    ncclResult_t getLocalBusCard(void *busCard);
    ncclResult_t setupVc(void *busCard, uint32_t *controlQp, std::vector<uint32_t> &dataQp);
    ncclResult_t progress();
    ncclResult_t processCqe(enum ibv_wc_opcode opcode, int qpNum, uint32_t immData);
    ncclResult_t isendCtrl(void *buf, void *ibRegElem, ctranIbRequest *req);
    ncclResult_t irecvCtrl(void **buf, struct ctranIbRemoteAccessKey *key, ctranIbRequest *req);
    ncclResult_t iput(const void *sbuf, void *dbuf, std::size_t len, void *ibRegElem,
        struct ctranIbRemoteAccessKey remoteAccessKey, bool notify, ctranIbRequest *req);
    ncclResult_t checkNotify(bool *notify);

    int peerRank;

  private:
    void setReady();
    ncclResult_t postRecvCtrlMsg(struct controlMsg *cmsg);
    ncclResult_t postSendCtrlMsg(struct controlMsg *cmsg);
    ncclResult_t postPutMsg(const void *sbuf, void *dbuf, std::size_t len,
        uint32_t lkey, uint32_t rkey, bool localNotify, bool notify);
    ncclResult_t postRecvNotifyMsg(int idx);

    struct ibv_qp *controlQp;
    std::vector<struct ibv_qp *> dataQp;

    struct {
      struct controlWr wr[MAX_CONTROL_MSGS];
      struct controlMsg cmsg[MAX_CONTROL_MSGS];
      struct ibv_mr *mr;
      std::deque<struct controlMsg *> freeMsg;
      std::deque<struct controlMsg *> postedMsg;
      std::deque<ctranIbRequest *> postedReq;
      std::deque<struct controlWr *> enqueuedWr;
    } sendCtrl;
    struct {
      struct controlWr wr[MAX_CONTROL_MSGS];
      struct controlMsg cmsg[MAX_CONTROL_MSGS];
      struct ibv_mr *mr;
      std::deque<struct controlMsg *> postedMsg;
      std::deque<struct controlWr *> unexWr;
      std::deque<struct controlWr *> enqueuedWr;
    } recvCtrl;
    struct {
      std::vector<std::deque<ctranIbRequest *>> postedWr;
    } put;

    bool isReady_;
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    int port;
    uint32_t maxMsgSize;
    std::mutex m;
    std::vector<std::deque<uint64_t>> notifications;
    std::unordered_map<int,int> qpNumToIdx;
};

#endif
