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

ncclResult_t ctranNvl::regMem(const void *buf, std::size_t len, void **nvlRegElem) {
  *nvlRegElem = nullptr;
  return ncclSuccess;
}

ncclResult_t ctranNvl::deregMem(const void *nvlRegElem) {
  return ncclSuccess;
}

ncclResult_t ctranNvl::isend(const void *buf, size_t len, int rank, const void *nvlRegElem, ctranNvlRequest **req) {
  struct ctranNvlElem *elem = new struct ctranNvlElem;
  elem->type = ctranNvlElem::elemType::ISEND;
  elem->u.isend.buf = buf;
  elem->u.isend.len = len;
  elem->req = new class ctranNvlRequest(const_cast<void *>(buf), len);
  *req = elem->req;

  this->pimpl->pendingSends.push_back(elem);

  return ncclSuccess;
}

ncclResult_t ctranNvl::irecv(void *buf, size_t len, int rank, const void *nvlRegElem, ctranNvlRequest **req) {
  struct ctranNvlElem *elem = new struct ctranNvlElem;
  elem->type = ctranNvlElem::elemType::IRECV;
  elem->u.irecv.buf = buf;
  elem->u.irecv.len = len;
  elem->req = new ctranNvlRequest(buf, len);
  *req = elem->req;

  this->pimpl->pendingRecvs.push_back(elem);

  return ncclSuccess;
}

ncclResult_t ctranNvl::progress(void) {
  ncclResult_t res = ncclSuccess;

  while (!this->pimpl->pendingRecvs.empty() && !this->pimpl->pendingSends.empty()) {
    auto r = this->pimpl->pendingRecvs.front();
    this->pimpl->pendingRecvs.erase(this->pimpl->pendingRecvs.begin());

    auto s = this->pimpl->pendingSends.front();
    this->pimpl->pendingSends.erase(this->pimpl->pendingSends.begin());

    if (r->u.irecv.len != s->u.isend.len) {
      WARN("CTRAN-NVL: unmatched send and recv\n");
      res = ncclSystemError;
      goto exit;
    }

    CUDACHECKGOTO(cudaMemcpyAsync(r->u.irecv.buf, s->u.isend.buf, r->u.irecv.len,
          cudaMemcpyDefault, this->pimpl->s), res, exit);
    CUDACHECKGOTO(cudaEventCreate(&r->e), res, exit);
    CUDACHECKGOTO(cudaEventRecord(r->e, this->pimpl->s), res, exit);

    this->pimpl->postedRecvs.push_back(r);
    this->pimpl->postedSends.push_back(s);
  }

  while (!this->pimpl->postedRecvs.empty() && !this->pimpl->postedSends.empty()) {
    auto cudaErr = cudaEventQuery(this->pimpl->postedRecvs.front()->e);
    if (cudaErr == cudaErrorNotReady) {
      break;
    } else if (cudaErr == cudaSuccess) {
      CUDACHECKGOTO(cudaEventDestroy(this->pimpl->postedRecvs.front()->e), res, exit);
      this->pimpl->postedRecvs.front()->req->complete();
      this->pimpl->postedRecvs.erase(this->pimpl->postedRecvs.begin());
      this->pimpl->postedSends.front()->req->complete();
      this->pimpl->postedSends.erase(this->pimpl->postedSends.begin());
    } else {
      WARN("CTRAN-NVL: cudaEventQuery returned error '%s'\n", cudaGetErrorString(cudaErr));
      res = ncclSystemError;
      goto exit;
    }
  }

exit:
  return res;
}

