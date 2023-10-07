// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include <vector>
#include <thread>
#include <unistd.h>
#include "nccl.h"
#include "checks.h"
#include "ctranIbBase.h"
#include "ctranIb.h"
#include "ctranIbImpl.h"
#include "ctranIbVc.h"

uint64_t ncclParamIbPkey();
uint64_t ncclParamIbGidIndex();
uint64_t ncclParamIbTc();
uint64_t ncclParamIbSl();
uint64_t ncclParamIbTimeout();
uint64_t ncclParamIbRetryCnt();

struct busCard {
  enum ibv_mtu mtu;
  uint64_t spn;
  uint64_t iid;
  uint32_t controlQpn;
  uint32_t dataQpn;
  uint8_t port;
};

ctranIb::impl::vc::vc(struct ibv_context *context, struct ibv_pd *pd, struct ibv_cq *cq,
    int port, int peerRank) {
  this->controlQp = nullptr;
  this->dataQp = nullptr;
  this->context = context;
  this->pd = pd;
  this->cq = cq;
  this->port = port;

  NCCLCHECKIGNORE(wrap_ibv_reg_mr(&this->control.send.mr, pd, (void *) this->control.send.controlMsgs,
        MAX_CONTROL_MSGS * sizeof(struct controlMsg),
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ));

  NCCLCHECKIGNORE(wrap_ibv_reg_mr(&this->control.recv.mr, pd, (void *) this->control.recv.controlMsgs,
        MAX_CONTROL_MSGS * sizeof(struct controlMsg),
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ));

  for (int i = 0; i < MAX_CONTROL_MSGS; i++) {
    this->control.send.wqeState[i].wqeType = wqeState::wqeType::CONTROL_SEND;
    this->control.send.wqeState[i].peerRank = peerRank;
    this->control.send.wqeState[i].peerWqe = &this->data.recv.wqeState[i];
    this->control.send.wqeState[i].u.control.cmsg = &this->control.send.controlMsgs[i];
    this->control.send.wqeState[i].wqeId = static_cast<uint64_t>(i);

    this->control.recv.wqeState[i].wqeType = wqeState::wqeType::CONTROL_RECV;
    this->control.recv.wqeState[i].peerRank = peerRank;
    this->control.recv.wqeState[i].peerWqe = &this->data.send.wqeState[i];
    this->control.recv.wqeState[i].u.control.cmsg = &this->control.recv.controlMsgs[i];
    this->control.recv.wqeState[i].wqeId = static_cast<uint64_t>(i);

    this->data.send.wqeState[i].wqeType = wqeState::wqeType::DATA_SEND;
    this->data.send.wqeState[i].peerRank = peerRank;
    this->data.send.wqeState[i].peerWqe = &this->control.recv.wqeState[i];
    this->data.send.wqeState[i].u.data.req = nullptr;
    this->data.send.wqeState[i].wqeId = static_cast<uint64_t>(i);

    this->data.recv.wqeState[i].wqeType = wqeState::wqeType::DATA_RECV;
    this->data.recv.wqeState[i].peerRank = peerRank;
    this->data.recv.wqeState[i].peerWqe = &this->control.send.wqeState[i];
    this->data.recv.wqeState[i].u.data.req = nullptr;
    this->data.recv.wqeState[i].wqeId = static_cast<uint64_t>(i);

    this->freeSendControlWqes.push_back(&this->control.send.wqeState[i]);
  }

  this->isReady_ = false;
  this->pendingSendQpWr = 0;
}

ctranIb::impl::vc::~vc() {
  NCCLCHECKIGNORE(wrap_ibv_dereg_mr(this->control.send.mr));
  NCCLCHECKIGNORE(wrap_ibv_dereg_mr(this->control.recv.mr));

  if (this->controlQp != nullptr) {
    /* we don't need to clean up the posted WQEs; destroying the QP
     * will automatically clear them */
    NCCLCHECKIGNORE(wrap_ibv_destroy_qp(this->controlQp));
    NCCLCHECKIGNORE(wrap_ibv_destroy_qp(this->dataQp));
  }
}

bool ctranIb::impl::vc::isReady() {
  bool r;

  this->m.lock();
  r = this->isReady_;
  this->m.unlock();

  return r;
}

void ctranIb::impl::vc::setReady() {
  this->m.lock();
  this->isReady_ = true;
  this->m.unlock();
}

std::size_t ctranIb::impl::vc::getBusCardSize() {
  return sizeof(struct busCard);
}

ncclResult_t ctranIb::impl::vc::getLocalBusCard(void *localBusCard) {
  ncclResult_t res = ncclSuccess;
  struct busCard *busCard = reinterpret_cast<struct busCard *>(localBusCard);

  struct ibv_port_attr portAttr;
  NCCLCHECKIGNORE(wrap_ibv_query_port(this->context, this->port, &portAttr));
  this->maxMsgSize = portAttr.max_msg_sz;

  /* create QP */
  struct ibv_qp_init_attr initAttr;
  memset(&initAttr, 0, sizeof(struct ibv_qp_init_attr));
  initAttr.send_cq = this->cq;
  initAttr.recv_cq = this->cq;
  initAttr.qp_type = IBV_QPT_RC;
  initAttr.sq_sig_all = 0;
  initAttr.cap.max_send_wr = MAX_SEND_WR;
  initAttr.cap.max_recv_wr = MAX_CONTROL_MSGS;
  initAttr.cap.max_send_sge = 1;
  initAttr.cap.max_recv_sge = 1;
  initAttr.cap.max_inline_data = 0;
  NCCLCHECKGOTO(wrap_ibv_create_qp(&this->controlQp, this->pd, &initAttr), res, exit);
  NCCLCHECKGOTO(wrap_ibv_create_qp(&this->dataQp, this->pd, &initAttr), res, exit);

  /* set QP to INIT state */
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = ncclParamIbPkey();
  qpAttr.port_num = this->port;
  qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
  NCCLCHECKGOTO(
    wrap_ibv_modify_qp(this->controlQp, &qpAttr,
                       IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
    res, exit);
  NCCLCHECKGOTO(
    wrap_ibv_modify_qp(this->dataQp, &qpAttr,
                       IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
    res, exit);

  /* create local business card */
  busCard->port = this->port;
  busCard->controlQpn = this->controlQp->qp_num;
  busCard->dataQpn = this->dataQp->qp_num;
  busCard->mtu = portAttr.active_mtu;

  union ibv_gid gid;
  NCCLCHECKGOTO(wrap_ibv_query_gid(this->context, this->port, ncclParamIbGidIndex(), &gid), res, exit);
  busCard->spn = gid.global.subnet_prefix;
  busCard->iid = gid.global.interface_id;

exit:
  return res;
}

ncclResult_t ctranIb::impl::vc::setupVc(void *busCard) {
  ncclResult_t res = ncclSuccess;
  struct ibv_qp_attr qpAttr;
  struct busCard *remoteBusCard = reinterpret_cast<struct busCard *>(busCard);

  /* set QP to RTR state */
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = remoteBusCard->mtu;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  qpAttr.ah_attr.is_global = 1;
  qpAttr.ah_attr.grh.dgid.global.subnet_prefix = remoteBusCard->spn;
  qpAttr.ah_attr.grh.dgid.global.interface_id = remoteBusCard->iid;
  qpAttr.ah_attr.grh.flow_label = 0;
  qpAttr.ah_attr.grh.sgid_index = ncclParamIbGidIndex();
  qpAttr.ah_attr.grh.hop_limit = 255;
  qpAttr.ah_attr.grh.traffic_class = ncclParamIbTc();
  qpAttr.ah_attr.sl = ncclParamIbSl();
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = remoteBusCard->port;

  qpAttr.dest_qp_num = remoteBusCard->controlQpn;
  NCCLCHECKGOTO(
    wrap_ibv_modify_qp(this->controlQp, &qpAttr,
                       IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                       IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER), res, exit);

  qpAttr.dest_qp_num = remoteBusCard->dataQpn;
  NCCLCHECKGOTO(
    wrap_ibv_modify_qp(this->dataQp, &qpAttr,
                       IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                       IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER), res, exit);

  /* set QP to RTS state */
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = ncclParamIbTimeout();
  qpAttr.retry_cnt = ncclParamIbRetryCnt();
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  NCCLCHECKGOTO(
    wrap_ibv_modify_qp(this->controlQp, &qpAttr,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                       IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC), res, exit);
  NCCLCHECKGOTO(
    wrap_ibv_modify_qp(this->dataQp, &qpAttr,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                       IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC), res, exit);

  /* post control WQEs */
  for (int i = 0; i < MAX_CONTROL_MSGS; i++) {
    NCCLCHECKGOTO(this->postRecvControlMsg(&this->control.recv.wqeState[i]), res, exit);
  }

  this->setReady();

exit:
  return res;
}

ncclResult_t ctranIb::impl::vc::postRecvControlMsg(struct wqeState *wqeState) {
  ncclResult_t res = ncclSuccess;


  struct ibv_sge sg;
  memset(&sg, 0, sizeof(sg));
  sg.addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wqeState->u.control.cmsg));
  sg.length = sizeof(struct controlMsg);
  sg.lkey = this->control.recv.mr->lkey;

  struct ibv_recv_wr wr, *badWr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wqeState));
  wr.next = nullptr;
  wr.sg_list = &sg;
  wr.num_sge = 1;
  NCCLCHECKGOTO(wrap_ibv_post_recv(this->controlQp, &wr, &badWr), res, exit);

exit:
  return res;
}

ncclResult_t ctranIb::impl::vc::postSendControlMsg(struct wqeState *wqeState) {
  ncclResult_t res = ncclSuccess;


  struct ibv_sge sg;
  memset(&sg, 0, sizeof(sg));
  sg.addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wqeState->u.control.cmsg));
  sg.length = sizeof(struct controlMsg);
  sg.lkey = this->control.send.mr->lkey;

  struct ibv_send_wr wr, *badWr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wqeState));
  wr.next = nullptr;
  wr.sg_list = &sg;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  NCCLCHECKGOTO(wrap_ibv_post_send(this->controlQp, &wr, &badWr), res, exit);

exit:
  return res;
}

ncclResult_t ctranIb::impl::vc::postRecvDataMsg(struct wqeState *wqeState) {
  ncclResult_t res = ncclSuccess;


  struct ibv_recv_wr wr, *badWr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wqeState));
  wr.next = nullptr;
  wr.num_sge = 0;
  NCCLCHECKGOTO(wrap_ibv_post_recv(this->dataQp, &wr, &badWr), res, exit);

exit:
  return res;
}

ncclResult_t ctranIb::impl::vc::postSendDataMsg(struct wqeState *wqeState, struct ibv_mr *mr) {
  ncclResult_t res = ncclSuccess;

  uint64_t offset = 0;
  uint64_t len = wqeState->u.data.req->len;

  while (len > 0) {
    uint64_t toSend = std::min(len, static_cast<uint64_t>(this->maxMsgSize));

    struct ibv_sge sg;
    memset(&sg, 0, sizeof(sg));
    sg.addr = reinterpret_cast<uint64_t>(wqeState->u.data.req->addr) + offset;
    sg.length = toSend;
    sg.lkey = mr->lkey;

    struct ibv_send_wr wr, *badWr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wqeState));
    wr.next = nullptr;
    wr.sg_list = &sg;
    wr.num_sge = 1;

    if (len > this->maxMsgSize) {
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.send_flags = 0;
    } else {
      wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      wr.send_flags = IBV_SEND_SIGNALED;
    }
    wr.wr.rdma.remote_addr = wqeState->u.data.remoteAddr + offset;
    wr.wr.rdma.rkey = wqeState->u.data.rkey;

    NCCLCHECKGOTO(wrap_ibv_post_send(this->dataQp, &wr, &badWr), res, exit);

    len -= toSend;
    offset += toSend;
    this->pendingSendQpWr++;
  }

exit:
  return res;
}

ncclResult_t ctranIb::impl::vc::progress(void) {
  ncclResult_t res = ncclSuccess;

  if (this->isReady() == false) {
    goto exit;
  }

  this->m.lock();

  for (auto qMap : this->data.commQueues) {
    auto commId = qMap.first;
    auto q = qMap.second;

    while (!q->recv.pendingQ.empty() && !this->freeSendControlWqes.empty()) {
      /* get a slot for a control message */
      auto controlWqeState = this->freeSendControlWqes.front();
      this->freeSendControlWqes.erase(this->freeSendControlWqes.begin());

      auto dataWqeState = controlWqeState->peerWqe;

      auto r = q->recv.pendingQ.front();
      q->recv.pendingQ.erase(q->recv.pendingQ.begin());

      struct ibv_mr *mr;
      mr = reinterpret_cast<struct ibv_mr *>(r->hdl);
      if (mr == nullptr) {
        WARN("CTRAN-IB: memory registration not found for addr %p", r->addr);
        res = ncclSystemError;
        goto exit;
      }

      /* post recv data WQE */
      dataWqeState->u.data.commId = commId;
      dataWqeState->u.data.remoteAddr = 0;
      dataWqeState->u.data.rkey = 0;
      dataWqeState->u.data.req = r;
      NCCLCHECKGOTO(this->postRecvDataMsg(dataWqeState), res, exit);
      q->recv.postedQ.push_back(dataWqeState);

      /* post send control WQE */
      auto cmsg = controlWqeState->u.control.cmsg;
      cmsg->u.msg.commId = commId;
      cmsg->u.msg.remoteAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(r->addr));
      cmsg->u.msg.rkey = mr->rkey;
      NCCLCHECKGOTO(this->postSendControlMsg(controlWqeState), res, exit);
    }

    while (!q->send.pendingQ.empty() && !q->rtrQ.empty()) {
      /* get the next pending send operation */
      auto r = q->send.pendingQ.front();
      int numSends = (r->len / this->maxMsgSize) + !!(r->len % this->maxMsgSize);
      if (this->pendingSendQpWr + numSends > MAX_SEND_WR) {
        break;
      }
      q->send.pendingQ.erase(q->send.pendingQ.begin());

      /* get the next RTR message */
      auto controlWqeState = q->rtrQ.front();
      q->rtrQ.erase(q->rtrQ.begin());

      auto dataWqeState = controlWqeState->peerWqe;

      struct ibv_mr *mr;
      mr = reinterpret_cast<struct ibv_mr *>(r->hdl);
      if (mr == nullptr) {
        WARN("CTRAN-IB: memory registration not found for addr %p", r->addr);
        res = ncclSystemError;
        goto exit;
      }

      /* post send data WQE */
      dataWqeState->u.data.commId = controlWqeState->u.control.cmsg->u.msg.commId;
      dataWqeState->u.data.remoteAddr = controlWqeState->u.control.cmsg->u.msg.remoteAddr;
      dataWqeState->u.data.rkey = controlWqeState->u.control.cmsg->u.msg.rkey;
      dataWqeState->u.data.req = r;
      NCCLCHECKGOTO(this->postSendDataMsg(dataWqeState, mr), res, exit);
      q->send.postedQ.push_back(dataWqeState);
      r->timestamp(ctranIbRequestTimestamp::GOT_RTR);
      r->timestamp(ctranIbRequestTimestamp::SEND_DATA_START);

      /* repost receive control WQE */
      NCCLCHECKGOTO(this->postRecvControlMsg(controlWqeState), res, exit);
    }
  }

exit:
  this->m.unlock();
  return res;
}

ncclResult_t ctranIb::impl::vc::processCqe(struct wqeState *wqeState) {
  ncclResult_t res = ncclSuccess;

  this->m.lock();

  switch (wqeState->wqeType) {
    case wqeState::wqeType::CONTROL_SEND:
      /* finished sending a control message */
      this->freeSendControlWqes.push_back(wqeState);
      break;

    case wqeState::wqeType::CONTROL_RECV:
      /* received a control message */
      {
        struct controlMsg *cmsg = wqeState->u.control.cmsg;
        uint64_t commId = cmsg->u.msg.commId;
        if (this->data.commQueues.find(commId) == this->data.commQueues.end()) {
          this->data.commQueues[commId] = new commQueues();
        }
        this->data.commQueues[cmsg->u.msg.commId]->rtrQ.push_back(wqeState);
      }
      break;

    case wqeState::wqeType::DATA_SEND:
      /* finished sending a data message */
      {
        auto q = this->data.commQueues[wqeState->u.data.commId];
        auto r = q->send.postedQ.front()->u.data.req;

        q->send.postedQ.front()->u.data.req->complete();
        q->send.postedQ.erase(q->send.postedQ.begin());

        int numSends = (r->len / this->maxMsgSize) + !!(r->len % this->maxMsgSize);
        this->pendingSendQpWr -= numSends;

        wqeState->u.data.req->timestamp(ctranIbRequestTimestamp::SEND_DATA_END);
      }
      break;

    case wqeState::wqeType::DATA_RECV:
      /* received a data message */
      {
        auto q = this->data.commQueues[wqeState->u.data.commId];
        q->recv.postedQ.front()->u.data.req->complete();
        q->recv.postedQ.erase(q->recv.postedQ.begin());
      }
      break;

    default:
      WARN("CTRAN-IB: Found unknown wqe type: %d", wqeState->wqeType);
      res = ncclSystemError;
      goto exit;
  }

exit:
  this->m.unlock();
  return res;
}

void ctranIb::impl::vc::enqueueIsend(ctranIbRequest *req, uint64_t commId) {
  this->m.lock();

  if (this->data.commQueues.find(commId) == this->data.commQueues.end()) {
    this->data.commQueues[commId] = new commQueues();
  }
  this->data.commQueues[commId]->send.pendingQ.push_back(req);

  req->timestamp(ctranIbRequestTimestamp::REQ_POSTED);

  this->m.unlock();
}

void ctranIb::impl::vc::enqueueIrecv(ctranIbRequest *req, uint64_t commId) {
  this->m.lock();

  if (this->data.commQueues.find(commId) == this->data.commQueues.end()) {
    this->data.commQueues[commId] = new commQueues();
  }
  this->data.commQueues[commId]->recv.pendingQ.push_back(req);

  this->m.unlock();
}
