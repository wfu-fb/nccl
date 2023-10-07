// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include <algorithm>
#include "nccl.h"
#include "ctranGraph.h"
#include "ctranGraphImpl.h"

ctranGraph::ctranGraph(ctranMapper *mapper) {
  this->pimpl = std::unique_ptr<impl>(new impl());
  this->pimpl->opHandleCounter = 0;
  this->pimpl->mapper = mapper;
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
  } else {
    *isComplete = false;
  }

exit:
  return res;
}
