// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include "ctranNvl.h"
#include "ctranNvlImpl.h"

ctranNvl::ctranNvl(ncclComm *comm) {
  this->pimpl = std::unique_ptr<impl>(new impl());
  CUDACHECKIGNORE(cudaStreamCreateWithFlags(&this->pimpl->s, cudaStreamNonBlocking));
}

ctranNvl::~ctranNvl() {
  CUDACHECKIGNORE(cudaStreamDestroy(this->pimpl->s));
}

ncclResult_t ctranNvl::regMem(const void *buf, std::size_t len, void **hdl) {
  *hdl = nullptr;
  return ncclSuccess;
}

ncclResult_t ctranNvl::deregMem(const void *hdl) {
  return ncclSuccess;
}

ncclResult_t ctranNvl::isend(const void *buf, size_t len, int rank, const void *hdl, uint64_t commId, ctranNvlRequest **req) {
  struct ctranNvlElem *elem = new struct ctranNvlElem;
  elem->type = ctranNvlElem::elemType::ISEND;
  elem->u.isend.buf = buf;
  elem->u.isend.len = len;
  elem->req = new class ctranNvlRequest(const_cast<void *>(buf), len, this);
  *req = elem->req;

  this->pimpl->m.lock();

  if (this->pimpl->ops.find(commId) == this->pimpl->ops.end()) {
    this->pimpl->ops[commId] = new struct ctranNvlCommQueues;
  }
  this->pimpl->ops[commId]->pendingSends.push_back(elem);

  this->pimpl->m.unlock();

  return ncclSuccess;
}

ncclResult_t ctranNvl::irecv(void *buf, size_t len, int rank, const void *hdl, uint64_t commId, ctranNvlRequest **req) {
  struct ctranNvlElem *elem = new struct ctranNvlElem;
  elem->type = ctranNvlElem::elemType::IRECV;
  elem->u.irecv.buf = buf;
  elem->u.irecv.len = len;
  elem->req = new ctranNvlRequest(buf, len, this);
  *req = elem->req;

  this->pimpl->m.lock();

  if (this->pimpl->ops.find(commId) == this->pimpl->ops.end()) {
    this->pimpl->ops[commId] = new struct ctranNvlCommQueues;
  }
  this->pimpl->ops[commId]->pendingRecvs.push_back(elem);

  this->pimpl->m.unlock();

  return ncclSuccess;
}

ncclResult_t ctranNvl::progress(void) {
  ncclResult_t res = ncclSuccess;

  this->pimpl->m.lock();

  for (auto cMap : this->pimpl->ops) {
    auto c = cMap.second;

    while (!c->pendingRecvs.empty() && !c->pendingSends.empty()) {
      auto r = c->pendingRecvs.front();
      c->pendingRecvs.erase(c->pendingRecvs.begin());

      auto s = c->pendingSends.front();
      c->pendingSends.erase(c->pendingSends.begin());

      if (r->u.irecv.len != s->u.isend.len) {
        WARN("CTRAN-NVL: unmatched send and recv\n");
        res = ncclSystemError;
        goto exit;
      }

      CUDACHECKGOTO(cudaMemcpyAsync(r->u.irecv.buf, s->u.isend.buf, r->u.irecv.len,
            cudaMemcpyDefault, this->pimpl->s), res, exit);
      CUDACHECKGOTO(cudaEventCreate(&r->e), res, exit);
      CUDACHECKGOTO(cudaEventRecord(r->e, this->pimpl->s), res, exit);

      c->postedRecvs.push_back(r);
      c->postedSends.push_back(s);
    }

    while (!c->postedRecvs.empty() && !c->postedSends.empty()) {
      auto cudaErr = cudaEventQuery(c->postedRecvs.front()->e);
      if (cudaErr == cudaErrorNotReady) {
        break;
      } else if (cudaErr == cudaSuccess) {
        CUDACHECKGOTO(cudaEventDestroy(c->postedRecvs.front()->e), res, exit);
        c->postedRecvs.front()->req->complete();
        c->postedRecvs.erase(c->postedRecvs.begin());
        c->postedSends.front()->req->complete();
        c->postedSends.erase(c->postedSends.begin());
      } else {
        WARN("CTRAN-NVL: cudaEventQuery returned error '%s'\n", cudaGetErrorString(cudaErr));
        res = ncclSystemError;
        goto exit;
      }
    }
  }

exit:
  this->pimpl->m.unlock();
  return res;
}

