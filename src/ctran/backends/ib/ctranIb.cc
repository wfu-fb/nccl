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

class roceHca {
public:
  roceHca(std::string hcaStr) {
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

ctranIb::ctranIb(ncclComm *comm) {
  this->pimpl = std::unique_ptr<impl>(new impl());

  this->pimpl->rank = comm->rank;
  this->pimpl->nRanks = comm->nRanks;
  std::vector<roceHca *> hcas;

  char* userIfsEnv = getenv("NCCL_IB_HCA");
  std::string s;
  if (userIfsEnv) {
    s = userIfsEnv;
  }
  std::string delim = ",";

  while (auto pos = s.find(delim)) {
    hcas.push_back(new roceHca(s.substr(0, pos)));
    s.erase(0, pos + delim.length());
    if (pos == std::string::npos) {
      break;
    }
  }

  NCCLCHECKIGNORE(wrap_ibv_symbols());

  struct ibv_device **devs;
  std::vector<struct ibv_device *> devices;
  std::vector<int> ports;
  int nDevs;
  NCCLCHECKIGNORE(wrap_ibv_get_device_list(&devs, &nDevs));

  for (int i = 0; i < nDevs; i++) {
    bool found = false;
    int port;
    for (auto d : hcas) {
      if (!strcmp(d->name.c_str(), devs[i]->name)) {
        found = true;
        port = d->port;
        break;
      }
    }
    if (!found) {
      continue;
    }

    struct ibv_device *device = devs[i];
    devices.push_back(device);
    ports.push_back(port);
  }

  if (devices.empty()) {
    throw std::bad_alloc();
  }

  int devId = comm->localRank % devices.size();
  this->pimpl->port = ports[devId];
  INFO(NCCL_INIT, "CTRAN-IB: using device %s, port %d", devices[devId]->name, this->pimpl->port);

  NCCLCHECKIGNORE(wrap_ibv_open_device(&this->pimpl->context, devices[devId]));
  NCCLCHECKIGNORE(wrap_ibv_alloc_pd(&this->pimpl->pd, this->pimpl->context));

  struct ibv_device_attr devAttr;
  NCCLCHECKIGNORE(wrap_ibv_query_device(this->pimpl->context, &devAttr));

  /* The max CQEs would not be enough for us in the worst case, where
   * we have a lot of VCs, and there is a lot of posted messages on
   * each of the VCs.  Static partitioning would reduce the number of
   * CQEs available to each VC in the common case.  Instead, we are
   * making an assumption here that the progress thread will pull out
   * completion entries fast enough that we will never overflow the
   * CQ. */
  NCCLCHECKIGNORE(wrap_ibv_create_cq(&this->pimpl->cq, this->pimpl->context,
        devAttr.max_cqe, nullptr, nullptr, 0));

  char ifName[MAX_IF_NAME_SIZE+1];
  union ncclSocketAddress ifAddr;
  int nIfs = ncclFindInterfaces(ifName, &ifAddr, MAX_IF_NAME_SIZE, 1);
  if (nIfs <= 0) {
    WARN("CTRAN-IB: no socket interfaces found\n");
  } else {
    INFO(NCCL_INIT, "CTRAN-IB: socket interface set to %s", ifName);
  }

  NCCLCHECKIGNORE(ncclSocketInit(&this->pimpl->listenSocket, &ifAddr));
  NCCLCHECKIGNORE(ncclSocketListen(&this->pimpl->listenSocket));

  this->pimpl->allListenSocketAddrs =
    static_cast<ncclSocketAddress *>(malloc(this->pimpl->nRanks * sizeof(ncclSocketAddress)));
  NCCLCHECKIGNORE(ncclSocketGetAddr(&this->pimpl->listenSocket,
        &this->pimpl->allListenSocketAddrs[this->pimpl->rank]));

  this->pimpl->listenThread = std::thread{ctranIb::impl::bootstrapAccept, this->pimpl.get()};

  bootstrapAllGather(comm->bootstrap, this->pimpl->allListenSocketAddrs, sizeof(ncclSocketAddress));

  for (int r = 0; r < this->pimpl->nRanks; r++) {
    this->pimpl->vcList.push_back(new ctranIb::impl::vc(this->pimpl->context, this->pimpl->pd,
          this->pimpl->cq, this->pimpl->port, r));
  }
}

ctranIb::~ctranIb(void) {
  NCCLCHECKIGNORE(this->pimpl->bootstrapTerminate());
  this->pimpl->listenThread.join();

  for (int r = 0; r < this->pimpl->nRanks; r++) {
    delete this->pimpl->vcList[r];
  }

  free(this->pimpl->allListenSocketAddrs);
  NCCLCHECKIGNORE(ncclSocketClose(&this->pimpl->listenSocket));

  NCCLCHECKIGNORE(wrap_ibv_destroy_cq(this->pimpl->cq));
  NCCLCHECKIGNORE(wrap_ibv_dealloc_pd(this->pimpl->pd));
  NCCLCHECKIGNORE(wrap_ibv_close_device(this->pimpl->context));
}

ncclResult_t ctranIb::regMem(const void *buf, std::size_t len, void **hdl) {
  ncclResult_t res = ncclSuccess;

  int pageSize = getpagesize();
  if (len < pageSize) {
    WARN("CTRAN-IB: cannot register buffer, size (%lu) smaller than page size (%d)", len, pageSize);
    res = ncclSystemError;
    goto exit;
  }

  struct ibv_mr *mr;
  NCCLCHECKGOTO(wrap_ibv_reg_mr(&mr, this->pimpl->pd, (void *) buf, len,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ), res, exit);
  *hdl = reinterpret_cast<void *>(mr);

exit:
  return res;
}

ncclResult_t ctranIb::deregMem(void *hdl) {
  struct ibv_mr *mr = reinterpret_cast<struct ibv_mr *>(hdl);
  NCCLCHECK(wrap_ibv_dereg_mr(mr));

  return ncclSuccess;
}

ncclResult_t ctranIb::progress(void) {
  ncclResult_t res = ncclSuccess;

  /* complete as many requests as possible */
  while (1) {
    struct ibv_wc wc;
    int count;

    this->pimpl->cqMutex.lock();
    res = wrap_ibv_poll_cq(this->pimpl->cq, 1, &wc, &count);
    this->pimpl->cqMutex.unlock();
    NCCLCHECKGOTO(res, res, exit);

    if (count == 0) {
      break;
    }

    /* wc.wr_id is valid even if the poll_cq returned an error; use it
     * to gather information about the error */
    auto wqeState = reinterpret_cast<struct wqeState *>(static_cast<uintptr_t>(wc.wr_id));
    int peerRank = wqeState->peerRank;
    auto vc = this->pimpl->vcList[peerRank];

    if (wc.status != IBV_WC_SUCCESS) {
      WARN("CTRAN-IB: wrap_ibv_poll_cq failed on op=%s, wqeId=%lu, peerRank=%d, with status=%d, '%s'",
          wqeName(wqeState->wqeType), wqeState->wqeId, peerRank, wc.status, this->pimpl->ibv_wc_status_str(wc.status));
      res = ncclSystemError;
      goto exit;
    }

    NCCLCHECKGOTO(vc->processCqe(wqeState), res, exit);
  }

  /* issue pending operations */
  for (int peerRank = 0; peerRank < this->pimpl->nRanks; peerRank++) {
    NCCLCHECKGOTO(this->pimpl->vcList[peerRank]->progress(), res, exit);
  }

exit:
  return res;
}

ncclResult_t ctranIb::isend(const void *buf, std::size_t len, int peerRank, void *hdl, uint64_t commId, ctranIbRequest **req) {
  ncclResult_t res = ncclSuccess;

  auto vc = this->pimpl->vcList[peerRank];
  if (this->pimpl->rank < peerRank && vc->isReady() == false) {
    NCCLCHECKGOTO(this->pimpl->bootstrapConnect(peerRank), res, exit);
  }

  *req = new ctranIbRequest(const_cast<void *>(buf), len, hdl, this);
  vc->enqueueIsend(*req, commId);

  NCCLCHECKGOTO(this->progress(), res, exit);

exit:
  return res;
}

ncclResult_t ctranIb::irecv(void *buf, std::size_t len, int peerRank, void *hdl, uint64_t commId, ctranIbRequest **req) {
  ncclResult_t res = ncclSuccess;

  auto vc = this->pimpl->vcList[peerRank];
  if (this->pimpl->rank < peerRank && vc->isReady() == false) {
    NCCLCHECKGOTO(this->pimpl->bootstrapConnect(peerRank), res, exit);
  }

  *req = new ctranIbRequest(const_cast<void *>(buf), len, hdl, this);
  vc->enqueueIrecv(*req, commId);

  NCCLCHECKGOTO(this->progress(), res, exit);

exit:
  return res;
}
