// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include <algorithm>
#include "nccl.h"
#include "param.h"
#include "ctranGraph.h"
#include "ctranGraphImpl.h"

NCCL_PARAM(CtranProfiling, "CTRAN_PROFILING", 0);

ctranGraph::ctranGraph(ctranMapper *mapper, std::string name) {
  this->pimpl = std::unique_ptr<impl>(new impl());
  this->pimpl->opHandleCounter = 0;
  this->pimpl->mapper = mapper;
  this->pimpl->name = name;
}

ctranGraph::~ctranGraph() {
  for (auto op : this->pimpl->allOps) {
    delete op;
  }
}

ncclResult_t ctranGraph::isend(const void *buf, std::size_t len, int rank,
    void *hdl, std::vector<int> deps, int *opHandle) {
  ncclResult_t res = ncclSuccess;

  *opHandle = this->pimpl->opHandleCounter++;

  struct ctranGraphElem *elem = new struct ctranGraphElem;
  elem->type = ctranGraphElem::ISEND;
  elem->u.isend.buf = buf;
  elem->u.isend.len = len;
  elem->u.isend.rank = rank;
  elem->u.isend.hdl = hdl;
  elem->req = nullptr;
  this->pimpl->allOps.push_back(elem);

  for (auto d : deps) {
    elem->upstreamDeps.push_back(this->pimpl->allOps[d]);
    this->pimpl->allOps[d]->downstreamDeps.push_back(elem);
  }

  if (elem->upstreamDeps.empty()) {
    this->pimpl->readyOps.push_back(elem);
  }

  return res;
}

ncclResult_t ctranGraph::irecv(void *buf, std::size_t len, int rank,
    void *hdl, std::vector<int> deps, int *opHandle) {
  ncclResult_t res = ncclSuccess;

  *opHandle = this->pimpl->opHandleCounter++;

  struct ctranGraphElem *elem = new struct ctranGraphElem;
  elem->type = ctranGraphElem::IRECV;
  elem->u.irecv.buf = buf;
  elem->u.irecv.len = len;
  elem->u.irecv.rank = rank;
  elem->u.irecv.hdl = hdl;
  elem->req = nullptr;
  this->pimpl->allOps.push_back(elem);

  for (auto d : deps) {
    elem->upstreamDeps.push_back(this->pimpl->allOps[d]);
    this->pimpl->allOps[d]->downstreamDeps.push_back(elem);
  }

  if (elem->upstreamDeps.empty()) {
    this->pimpl->readyOps.push_back(elem);
  }

  return res;
}

ncclResult_t ctranGraph::icopy(void *dbuf, const void *sbuf, std::size_t len, std::vector<int> deps,
    int *opHandle) {
  ncclResult_t res = ncclSuccess;

  *opHandle = this->pimpl->opHandleCounter++;

  struct ctranGraphElem *elem = new struct ctranGraphElem;
  elem->type = ctranGraphElem::ICOPY;
  elem->u.icopy.sbuf = sbuf;
  elem->u.icopy.dbuf = dbuf;
  elem->u.icopy.len = len;
  elem->req = nullptr;
  this->pimpl->allOps.push_back(elem);

  for (auto d : deps) {
    elem->upstreamDeps.push_back(this->pimpl->allOps[d]);
    this->pimpl->allOps[d]->downstreamDeps.push_back(elem);
  }

  if (elem->upstreamDeps.empty()) {
    this->pimpl->readyOps.push_back(elem);
  }

  return res;
}

ncclResult_t ctranGraph::test(bool *isComplete) {
  ncclResult_t res = ncclSuccess;
  bool reqComplete;

  std::vector<struct ctranGraphElem *> tmp;
  for (auto op : this->pimpl->postedOps) {
    NCCLCHECKGOTO(op->req->test(&reqComplete), res, exit);
    if (reqComplete) {
      if (ncclParamCtranProfiling() && op->type == ctranGraphElem::ISEND &&
          op->u.isend.rank != this->pimpl->mapper->rank) {
        struct timestamp t;
        t.waitTime = op->req->getWaitTime();
        t.commTime = op->req->getCommTime();
        if (op->type == ctranGraphElem::ISEND) {
          t.peer = op->u.isend.rank;
          t.len = op->u.isend.len;
        } else {
          t.peer = op->u.irecv.rank;
          t.len = op->u.irecv.len;
        }
        this->pimpl->timestamps.push_back(t);
      }

      for (auto d : op->downstreamDeps) {
        d->upstreamDeps.erase(std::remove(d->upstreamDeps.begin(), d->upstreamDeps.end(), op), d->upstreamDeps.end());
        if (d->upstreamDeps.empty()) {
          this->pimpl->readyOps.push_back(d);
        }
      }
    } else {
      tmp.push_back(op);
    }
  }
  this->pimpl->postedOps.clear();
  this->pimpl->postedOps = tmp;
  tmp.clear();

  for (auto op : this->pimpl->readyOps) {
    switch (op->type) {
      case ctranGraphElem::ISEND:
        NCCLCHECKGOTO(this->pimpl->mapper->isend(op->u.isend.buf, op->u.isend.len, op->u.isend.rank, op->u.isend.hdl, &op->req), res, exit);
        break;

      case ctranGraphElem::IRECV:
        NCCLCHECKGOTO(this->pimpl->mapper->irecv(op->u.irecv.buf, op->u.irecv.len, op->u.irecv.rank, op->u.irecv.hdl, &op->req), res, exit);
        break;

      case ctranGraphElem::ICOPY:
        NCCLCHECKGOTO(this->pimpl->mapper->icopy(op->u.icopy.dbuf, op->u.icopy.sbuf, op->u.icopy.len, &op->req), res, exit);
        break;

      default:
        res = ncclSystemError;
        goto exit;
    }

    this->pimpl->postedOps.push_back(op);
  }
  this->pimpl->readyOps.clear();

  if (this->pimpl->postedOps.empty() && this->pimpl->readyOps.empty()) {
    *isComplete = true;

    if (ncclParamCtranProfiling()) {
      for (auto t : this->pimpl->timestamps) {
        std::cout << "CTRAN-GRAPH: "
          << "coll=" << this->pimpl->name
          << "; collId=" << this->pimpl->mapper->collId
          << "; rank=" << this->pimpl->mapper->rank
          << "; peer=" << t.peer
          << "; comm=" << this->pimpl->mapper->commHash
          << "; len=" << t.len
          << "; waitTime(us)=" << t.waitTime
          << "; commTime(us)=" << t.commTime
          << "; commBW(GB/s)=" << static_cast<double>(t.len) / ((t.waitTime + t.commTime) * 1000.0)
          << std::endl << std::flush;
      }
      this->pimpl->mapper->collId++;
    }
  } else {
    *isComplete = false;
  }

exit:
  return res;
}
