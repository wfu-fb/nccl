// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstdio>
#include <iostream>
#include <vector>
#include <thread>
#include <unistd.h>
#include "nccl.h"
#include "checks.h"
#include "nccl_cvars.h"
#include "CtranIbBase.h"
#include "CtranIb.h"
#include "CtranIbImpl.h"
#include "CtranIbVc.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_IB_HCA
   type        : stringlist
   default     :
   description : |-
     List of IB HCAs available for NCCL to use.
     (this needs to be renamed to NCCL_IB_HCA_LIST eventually)

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

class RoceHca {
public:
  RoceHca(std::string hcaStr) {
    std::string s = hcaStr;
    std::string delim = ":";

    auto pos = s.find(delim);
    if (pos == std::string::npos) {
      this->name = s;
      this->port = 0;
    } else {
      this->name = s.substr(0, pos);
      s.erase(0, pos + delim.length());
      this->port = std::stoi(s);
    }
  }

  std::string name;
  int port;
};

CtranIbSingleton &CtranIbSingleton::getInstance(void) {
  static CtranIbSingleton s;
  return s;
}

CtranIbSingleton::CtranIbSingleton(void) {
  std::vector<RoceHca> hcas;
  // Avoid copy triggered by resize
  hcas.reserve(NCCL_IB_HCA.size());

  for (auto hca : NCCL_IB_HCA) {
    // Copy value to each vector element so it can be freed automatically
    hcas.push_back(RoceHca(hca));
  }

  NCCLCHECKIGNORE(wrap_ibv_symbols());

  struct ibv_device **devs;
  std::vector<struct ibv_device *> devices;
  int nDevs;
  NCCLCHECKIGNORE(wrap_ibv_get_device_list(&devs, &nDevs));

  for (int i = 0; i < nDevs; i++) {
    bool found = false;
    int port;
    for (auto d : hcas) {
      if (!strcmp(d.name.c_str(), devs[i]->name)) {
        found = true;
        port = d.port;
        break;
      }
    }
    if (!found) {
      continue;
    }

    struct ibv_device *device = devs[i];
    devices.push_back(device);
    this->ports.push_back(port);
  }

  if (devices.empty()) {
    throw std::bad_alloc();
  }

  for (auto i = 0; i < devices.size(); i++) {
    struct ibv_context *context;
    struct ibv_pd *pd;
    NCCLCHECKIGNORE(wrap_ibv_open_device(&context, devices[i]));
    NCCLCHECKIGNORE(wrap_ibv_alloc_pd(&pd, context));

    this->contexts.push_back(context);
    this->pds.push_back(pd);
    this->devNames.push_back(devices[i]->name);
  }
}

CtranIbSingleton::~CtranIbSingleton() {
  for (auto pd : this->pds) {
    NCCLCHECKIGNORE(wrap_ibv_dealloc_pd(pd));
  }

  for (auto context : this->contexts) {
    NCCLCHECKIGNORE(wrap_ibv_close_device(context));
  }
}

CtranIb::CtranIb(ncclComm *comm) {
  this->pimpl_ = std::unique_ptr<Impl>(new Impl());

  this->pimpl_->rank = comm->rank;
  this->pimpl_->nRanks = comm->nRanks;

  CtranIbSingleton& s = CtranIbSingleton::getInstance();

  this->pimpl_->context = s.contexts[comm->cudaDev];
  this->pimpl_->pd = s.pds[comm->cudaDev];
  this->pimpl_->port = s.ports[comm->cudaDev];
  INFO(NCCL_INIT, "CTRAN-IB: using device %s, port %d commHash %lu", s.devNames[comm->cudaDev].c_str(), this->pimpl_->port, comm->commHash);

  struct ibv_device_attr devAttr;
  NCCLCHECKIGNORE(wrap_ibv_query_device(this->pimpl_->context, &devAttr));

  /* The max CQEs would not be enough for us in the worst case, where
   * we have a lot of VCs, and there is a lot of posted messages on
   * each of the VCs.  Static partitioning would reduce the number of
   * CQEs available to each VC in the common case.  Instead, we are
   * making an assumption here that the progress thread will pull out
   * completion entries fast enough that we will never overflow the
   * CQ. */
  NCCLCHECKIGNORE(wrap_ibv_create_cq(&this->pimpl_->cq, this->pimpl_->context,
        devAttr.max_cqe, nullptr, nullptr, 0));

  // Locally initialize all virtual connections before launching the listen
  // thread
  for (int r = 0; r < this->pimpl_->nRanks; r++) {
    this->pimpl_->vcList.push_back(new CtranIb::Impl::VirtualConn(this->pimpl_->context, this->pimpl_->pd,
          this->pimpl_->cq, this->pimpl_->port, r));
    this->pimpl_->numUnsignaledPuts.push_back(0);
  }

  char ifName[MAX_IF_NAME_SIZE+1];
  union ncclSocketAddress ifAddr;
  int nIfs = ncclFindInterfaces(ifName, &ifAddr, MAX_IF_NAME_SIZE, 1);
  if (nIfs <= 0) {
    WARN("CTRAN-IB: no socket interfaces found\n");
  } else {
    INFO(NCCL_INIT, "CTRAN-IB: socket interface set to %s", ifName);
  }

  NCCLCHECKIGNORE(ncclSocketInit(&this->pimpl_->listenSocket, &ifAddr));
  NCCLCHECKIGNORE(ncclSocketListen(&this->pimpl_->listenSocket));

  this->pimpl_->allListenSocketAddrs =
    static_cast<ncclSocketAddress *>(malloc(this->pimpl_->nRanks * sizeof(ncclSocketAddress)));
  NCCLCHECKIGNORE(ncclSocketGetAddr(&this->pimpl_->listenSocket,
        &this->pimpl_->allListenSocketAddrs[this->pimpl_->rank]));

  bootstrapAllGather(comm->bootstrap, this->pimpl_->allListenSocketAddrs, sizeof(ncclSocketAddress));

  this->pimpl_->listenThread = std::thread{CtranIb::Impl::bootstrapAccept, this->pimpl_.get()};
}

CtranIb::~CtranIb(void) {
  NCCLCHECKIGNORE(this->pimpl_->bootstrapTerminate());
  this->pimpl_->listenThread.join();

  for (int r = 0; r < this->pimpl_->nRanks; r++) {
    delete this->pimpl_->vcList[r];
  }

  free(this->pimpl_->allListenSocketAddrs);
  NCCLCHECKIGNORE(ncclSocketClose(&this->pimpl_->listenSocket));

  NCCLCHECKIGNORE(wrap_ibv_destroy_cq(this->pimpl_->cq));
}

ncclResult_t CtranIb::regMem(const void *buf, std::size_t len, void **ibRegElem) {
  ncclResult_t res = ncclSuccess;

  int pageSize = getpagesize();
  if (len <= pageSize) {
    WARN("CTRAN-IB: cannot register buffer, size (%lu) <= page size (%d)", len, pageSize);
    res = ncclSystemError;
    goto exit;
  }

  struct ibv_mr *mr;
  NCCLCHECKGOTO(wrap_ibv_reg_mr(&mr, this->pimpl_->pd, (void *) buf, len,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ), res, exit);
  *ibRegElem = reinterpret_cast<void *>(mr);

exit:
  return res;
}

ncclResult_t CtranIb::deregMem(void *ibRegElem) {
  ncclResult_t res = ncclSuccess;

  struct ibv_mr *mr = reinterpret_cast<struct ibv_mr *>(ibRegElem);
  NCCLCHECKGOTO(wrap_ibv_dereg_mr(mr), res, exit);

exit:
  return res;
}

ncclResult_t CtranIb::progress(void) {
  ncclResult_t res = ncclSuccess;

  /* complete as many requests as possible */
  while (1) {
    struct ibv_wc wc;
    int count;

    res = wrap_ibv_poll_cq(this->pimpl_->cq, 1, &wc, &count);
    NCCLCHECKGOTO(res, res, exit);

    if (count == 0) {
      break;
    }

    /* wc.wr_id is valid even if the poll_cq returned an error; use it
     * to gather information about the error */
    this->pimpl_->m.lock();
    auto vc = this->pimpl_->vcList[this->pimpl_->qpToRank[wc.qp_num]];
    this->pimpl_->m.unlock();

    if (wc.status != IBV_WC_SUCCESS) {
      WARN("CTRAN-IB: wrap_ibv_poll_cq failed, peerRank=%d, with status=%d, '%s'",
          vc->peerRank, wc.status, this->pimpl_->ibv_wc_status_str(wc.status));
      res = ncclSystemError;
      goto exit;
    }

    NCCLCHECKGOTO(vc->processCqe(wc.opcode, wc.qp_num, wc.imm_data), res, exit);
  }

  /* we should have pendingOps only if the connection was not
   * established yet.  The below algorithm is a bit inefficient, but
   * that is OK as it should not happen in the critical path. */
  if (!this->pimpl_->pendingOps.empty()) {
    std::vector<int> peerRanks;
    std::vector<struct PendingOp *> tmp = this->pimpl_->pendingOps;
    this->pimpl_->pendingOps.clear();

    for (auto op : tmp) {
      int rank = (op->type == PendingOp::PendingOpType::ISEND_CTRL) ? op->isendCtrl.peerRank :
        op->irecvCtrl.peerRank;

      /* if we already encounted this peer, skip all operations to the
       * same peer; otherwise we might end up sending messages out of
       * order */
      if (std::find(peerRanks.begin(), peerRanks.end(), rank) != peerRanks.end()) {
        this->pimpl_->pendingOps.push_back(op);
        continue;
      }

      auto vc = this->pimpl_->vcList[rank];
      if (op->type == PendingOp::PendingOpType::ISEND_CTRL) {
        if (vc->isReady() == true) {
          NCCLCHECKGOTO(vc->isendCtrl(op->isendCtrl.buf, op->isendCtrl.ibRegElem, op->isendCtrl.req), res, exit);
          delete op;
        } else {
          this->pimpl_->pendingOps.push_back(op);
          peerRanks.push_back(rank);
        }
      } else {
        if (vc->isReady() == true) {
          NCCLCHECKGOTO(vc->irecvCtrl(op->irecvCtrl.buf, op->irecvCtrl.key, op->irecvCtrl.req), res, exit);
          delete op;
        } else {
          this->pimpl_->pendingOps.push_back(op);
          peerRanks.push_back(rank);
        }
      }
    }
  }

exit:
  return res;
}

ncclResult_t CtranIb::isendCtrl(void *buf, void *ibRegElem, int peerRank, CtranIbRequest **req) {
  ncclResult_t res = ncclSuccess;

  auto vc = this->pimpl_->vcList[peerRank];
  if (this->pimpl_->rank < peerRank && vc->isReady() == false) {
    NCCLCHECKGOTO(this->pimpl_->bootstrapConnect(peerRank), res, exit);
  }

  *req = new CtranIbRequest();
  if (vc->isReady() == true) {
    NCCLCHECKGOTO(vc->isendCtrl(buf, ibRegElem, *req), res, exit);
  } else {
    auto pendingOp = new struct PendingOp;
    pendingOp->type = PendingOp::PendingOpType::ISEND_CTRL;
    pendingOp->isendCtrl.buf = buf;
    pendingOp->isendCtrl.ibRegElem = ibRegElem;
    pendingOp->isendCtrl.peerRank = peerRank;
    pendingOp->isendCtrl.req = *req;
    this->pimpl_->pendingOps.push_back(pendingOp);
  }

exit:
  return res;
}

ncclResult_t CtranIb::irecvCtrl(void **buf, struct CtranIbRemoteAccessKey *key, int peerRank,
    CtranIbRequest **req) {
  ncclResult_t res = ncclSuccess;

  auto vc = this->pimpl_->vcList[peerRank];
  if (this->pimpl_->rank < peerRank && vc->isReady() == false) {
    NCCLCHECKGOTO(this->pimpl_->bootstrapConnect(peerRank), res, exit);
  }

  *req = new CtranIbRequest();
  if (vc->isReady() == true) {
    NCCLCHECKGOTO(vc->irecvCtrl(buf, key, *req), res, exit);
  } else {
    auto pendingOp = new struct PendingOp;
    pendingOp->type = PendingOp::PendingOpType::IRECV_CTRL;
    pendingOp->irecvCtrl.buf = buf;
    pendingOp->irecvCtrl.key = key;
    pendingOp->irecvCtrl.peerRank = peerRank;
    pendingOp->irecvCtrl.req = *req;
    this->pimpl_->pendingOps.push_back(pendingOp);
  }

exit:
  return res;
}

ncclResult_t CtranIb::iput(const void *sbuf, void *dbuf, std::size_t len, int peerRank, void *ibRegElem,
    struct CtranIbRemoteAccessKey remoteAccessKey, bool notify, CtranIbRequest **req) {
  ncclResult_t res = ncclSuccess;
  CtranIbRequest *r = nullptr;

  if (req != nullptr) {
    *req = new CtranIbRequest();
    r = *req;
    this->pimpl_->numUnsignaledPuts[peerRank] = 0;
  } else {
    this->pimpl_->numUnsignaledPuts[peerRank]++;
    if (this->pimpl_->numUnsignaledPuts[peerRank] == MAX_SEND_WR) {
      r = &this->pimpl_->fakeReq;
      this->pimpl_->numUnsignaledPuts[peerRank] = 0;
    }
  }

  NCCLCHECKGOTO(this->pimpl_->vcList[peerRank]->iput(sbuf, dbuf, len, ibRegElem, remoteAccessKey, notify, r),
      res, exit);

exit:
  return res;
}

ncclResult_t CtranIb::checkNotify(int peerRank, bool *notify) {
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(this->progress(), res, exit);
  *notify = this->pimpl_->vcList[peerRank]->checkNotify();

exit:
  return res;
}
