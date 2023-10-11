// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_VC_H_
#define CTRAN_IB_VC_H_

#include <vector>
#include <unordered_map>
#include "ibvwrap.h"
#include "ctranIbBase.h"

#define MAX_CONTROL_MSGS (128)
#define MAX_SEND_WR      (256)

class ctranIb::impl::vc {
  public:
    vc(struct ibv_context *context, struct ibv_pd *pd, struct ibv_cq *cq,
        int port, int peerRank);
    ~vc();

    bool isReady();
    std::size_t getBusCardSize();
    ncclResult_t getLocalBusCard(void *busCard);
    ncclResult_t setupVc(void *busCard);
    ncclResult_t progress();
    ncclResult_t processCqe(struct wqeState *wqeState);
    void enqueueIsend(ctranIbRequest *req);
    void enqueueIrecv(ctranIbRequest *req);

  private:
    void setReady();
    ncclResult_t postRecvControlMsg(struct wqeState *wqeState);
    ncclResult_t postSendControlMsg(struct wqeState *wqeState);
    ncclResult_t postRecvDataMsg(struct wqeState *wqeState);
    ncclResult_t postSendDataMsg(struct wqeState *wqeState, struct ibv_mr *mr);

    struct ibv_qp *controlQp;
    struct ibv_qp *dataQp;

    struct {
      struct {
        struct wqeState wqeState[MAX_CONTROL_MSGS];
        struct controlMsg controlMsgs[MAX_CONTROL_MSGS];
        struct ibv_mr *mr;
      } send, recv;
    } control;

    struct {
      struct {
        struct wqeState wqeState[MAX_CONTROL_MSGS];
        std::vector<struct wqeState *> postedQ;
        std::vector<ctranIbRequest *> pendingQ;
      } send, recv;
      std::vector<struct wqeState *> rtrQ;
    } data;

    std::vector<struct wqeState *> freeSendControlWqes;

    bool isReady_;
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    int port;
    uint32_t maxMsgSize;
    int pendingSendQpWr;
};

#endif
