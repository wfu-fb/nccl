// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CtranMapper.h"
#include <unistd.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include "CtranMapperImpl.h"
#include "comm.h"
#include "nccl_cvars.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_CTRAN_REGISTER
   type        : enum
   default     : lazy
   choices     : none, lazy, eager
   description : |-
     Kind of registration to use for ctran user buffers
     none - No registration
     lazy - Lazy registration (keep track of user-provided registration
            buffers, but delay the actual registration till the buffer
            is used for a communication operation)
     eager - Eager registration (register buffers as soon as it is
             provided by the user)

 - name        : NCCL_CTRAN_BACKENDS
   type        : enumlist
   default     : ib
   choices     : ib
   description : |-
     Backends to enable for ctran
     ib - RoCE/IB backend

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

CtranMapper::CtranMapper(ncclComm* comm) {
  this->pimpl_ = std::unique_ptr<impl>(new impl());

  /* mapperRegElemList */
  this->pimpl_->mapperRegElemList =
      std::unique_ptr<class CtranAvlTree>(new class CtranAvlTree());

  /* check user preference for backends */
  for (auto b : NCCL_CTRAN_BACKENDS) {
    if (b == NCCL_CTRAN_BACKENDS::ib) {
      this->pimpl_->backends.push_back(CtranMapperBackend::IB);
    }
  }

  /* enable available backends
   * NOTE: currently only support IB backend
   */
  this->pimpl_->ctranIb = nullptr;
  auto it = std::find(
      this->pimpl_->backends.begin(),
      this->pimpl_->backends.end(),
      CtranMapperBackend::IB);
  /* initialize Ctran IB backend */
  if (it != this->pimpl_->backends.end()) {
    try {
      this->pimpl_->ctranIb =
          std::unique_ptr<class CtranIb>(new class CtranIb(comm));
    } catch (const std::bad_alloc& e) {
      WARN("CTRAN: IB backend not enabled");
    }
  }

  /* create rankBackendMap, index 'i' indicates the backend used for rank 'i' */
  for (int i = 0; i < comm->nRanks; i++) {
    if (this->pimpl_->ctranIb != nullptr) {
      this->pimpl_->rankBackendMap.push_back(CtranMapperBackend::IB);
    } else {
      this->pimpl_->rankBackendMap.push_back(CtranMapperBackend::UNSET);
    }
  }

  CUDACHECKIGNORE(
      cudaStreamCreateWithFlags(&this->internalStream, cudaStreamNonBlocking));

  this->rank = comm->rank;
  this->commHash = comm->commHash;
}

CtranMapper::~CtranMapper() {
  /* safely de-register any bufferes applications may miss */
  auto v = this->pimpl_->mapperRegElemList->getAllElems();
  for (auto hdl : v) {
    NCCLCHECKIGNORE(this->deregMem(hdl));
  }

  CUDACHECKIGNORE(cudaStreamDestroy(this->internalStream));
}

ncclResult_t CtranMapper::impl::regMem(
    struct CtranMapperRegElem* mapperRegElem) {
  ncclResult_t res = ncclSuccess;

  if (this->ctranIb != nullptr) {
    assert(mapperRegElem->ibRegElem == nullptr);
    NCCLCHECKGOTO(
        this->ctranIb->regMem(
            mapperRegElem->buf, mapperRegElem->len, &mapperRegElem->ibRegElem),
        res,
        exit);
  }

  mapperRegElem->state = CtranMapperRegElemState::REGISTERED;

  INFO(
      NCCL_COLL,
      "CTRAN-MAPPER: registered buffer %p len %ld, state %d",
      mapperRegElem->buf,
      mapperRegElem->len,
      mapperRegElem->state);

exit:
  return res;
}

ncclResult_t CtranMapper::impl::deregMem(
    struct CtranMapperRegElem* mapperRegElem) {
  ncclResult_t res = ncclSuccess;

  if (this->ctranIb != nullptr) {
    NCCLCHECKGOTO(this->ctranIb->deregMem(mapperRegElem->ibRegElem), res, exit);
  }

  INFO(
      NCCL_COLL,
      "CTRAN-MAPPER: deregister buffer %p len %ld, state %d",
      mapperRegElem->buf,
      mapperRegElem->len,
      mapperRegElem->state);

exit:
  return res;
}

ncclResult_t CtranMapper::regMem(
    const void* buf,
    std::size_t len,
    void** hdl,
    bool forceRegist) {
  ncclResult_t res = ncclSuccess;
  struct CtranMapperRegElem* mapperRegElem = nullptr;

  auto hdl_ = this->pimpl_->mapperRegElemList->search(buf, len);
  if (hdl_) {
    *hdl = hdl_;
    goto exit;
  }

  cudaPointerAttributes attr;
  CUDACHECKGOTO(cudaPointerGetAttributes(&attr, buf), res, exit);
  if (attr.type != cudaMemoryTypeDevice) {
    WARN("CTRAN-MAPPER: buf %p is not a device buffer\n", buf);
    res = ncclSystemError;
    goto exit;
  }

  /* create a new entry to cache the buffer info in the AVL tree */
  mapperRegElem = new struct CtranMapperRegElem;
  mapperRegElem->buf = buf;
  mapperRegElem->len = len;
  mapperRegElem->ibRegElem = nullptr;
  mapperRegElem->state = CtranMapperRegElemState::CACHED;

  *hdl = this->pimpl_->mapperRegElemList->insert(
      buf, len, reinterpret_cast<void*>(mapperRegElem));

  /* regiser the buffer only if on Eager mode or forced by caller */
  if (NCCL_CTRAN_REGISTER == NCCL_CTRAN_REGISTER::eager || forceRegist) {
    NCCLCHECKGOTO(this->pimpl_->regMem(mapperRegElem), res, fail);
  }

exit:
  return res;
fail:
  if (*hdl) {
    this->pimpl_->mapperRegElemList->remove(*hdl);
  }
  delete mapperRegElem;
  goto exit;
}

ncclResult_t CtranMapper::deregMem(void* hdl) {
  ncclResult_t res = ncclSuccess;
  struct CtranMapperRegElem* mapperRegElem = nullptr;

  /* fast return for invalid handle: nullptr or cannot be found in the cache */
  if (hdl == nullptr ||
      !(mapperRegElem = reinterpret_cast<struct CtranMapperRegElem*>(
            this->pimpl_->mapperRegElemList->lookup(hdl)))) {
    return ncclSuccess;
  }

  if (mapperRegElem->state == CtranMapperRegElemState::REGISTERED) {
    NCCLCHECKGOTO(this->pimpl_->deregMem(mapperRegElem), res, exit);
  }

exit:
  return this->pimpl_->mapperRegElemList->remove(hdl);
}

ncclResult_t CtranMapper::searchRegHandle(
    const void* buf,
    std::size_t len,
    void** hdl,
    bool* dynamicRegist) {
  ncclResult_t res = ncclSuccess;

  *hdl = this->pimpl_->mapperRegElemList->search(buf, len);

  if (*hdl != nullptr) {
    struct CtranMapperRegElem* mapperRegElem =
        reinterpret_cast<struct CtranMapperRegElem*>(
            this->pimpl_->mapperRegElemList->lookup(*hdl));

    // User has cached it but we delay the registration until now due to lazy
    // registration
    if (mapperRegElem->state == CtranMapperRegElemState::CACHED) {
      NCCLCHECKGOTO(this->pimpl_->regMem(mapperRegElem), res, exit);
    }
    *dynamicRegist = false;
  } else {
    // Oops, the buffer is not cached nor registered by user. Thus, we have to
    // register it on demand
    NCCLCHECKGOTO(
        this->regMem(buf, len, hdl, true /* force register */), res, exit);
    // caller is responsible for deregisgration
    *dynamicRegist = true;
  }

exit:
  return res;
}

ncclResult_t CtranMapper::icopy(
    void* dbuf,
    const void* sbuf,
    std::size_t len,
    CtranMapperRequest** req) {
  ncclResult_t res = ncclSuccess;

  *req = new CtranMapperRequest(this);
  CUDACHECKGOTO(
      cudaMemcpyAsync(dbuf, sbuf, len, cudaMemcpyDefault, this->internalStream),
      res,
      exit);

exit:
  return res;
}

ncclResult_t CtranMapper::progress(void) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl_->ctranIb != nullptr) {
    NCCLCHECKGOTO(this->pimpl_->ctranIb->progress(), res, exit);
  }

exit:
  return res;
}

ncclResult_t CtranMapper::isendCtrl(
    void* buf,
    void* hdl,
    int rank,
    CtranMapperRequest** req) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl_->ctranIb != nullptr) {
    struct CtranMapperRegElem* mapperRegElem =
        reinterpret_cast<struct CtranMapperRegElem*>(
            this->pimpl_->mapperRegElemList->lookup(hdl));

    CtranIbRequest** ibReqPtr = nullptr;
    if (req) {
      *req = new CtranMapperRequest(this);
      ibReqPtr = &((*req)->ibReq);
    }
    res = this->pimpl_->ctranIb->isendCtrl(
        buf, mapperRegElem->ibRegElem, rank, ibReqPtr);
  }

  return res;
}

ncclResult_t CtranMapper::irecvCtrl(
    void** buf,
    struct CtranMapperRemoteAccessKey* key,
    int rank,
    CtranMapperRequest** req) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl_->ctranIb != nullptr) {
    CtranIbRequest** ibReqPtr = nullptr;
    if (req) {
      *req = new CtranMapperRequest(this);
      ibReqPtr = &((*req)->ibReq);
    }
    res = this->pimpl_->ctranIb->irecvCtrl(buf, &key->ibKey, rank, ibReqPtr);
  }

  return res;
}

ncclResult_t CtranMapper::iput(
    const void* sbuf,
    void* dbuf,
    std::size_t len,
    int rank,
    void* shdl,
    struct CtranMapperRemoteAccessKey remoteAccessKey,
    bool notify,
    CtranMapperRequest** req) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl_->ctranIb != nullptr) {
    struct CtranMapperRegElem* mapperRegElem =
        reinterpret_cast<struct CtranMapperRegElem*>(
            this->pimpl_->mapperRegElemList->lookup(shdl));
    CtranIbRequest** ibReqPtr = nullptr;
    if (req) {
      *req = new CtranMapperRequest(this);
      ibReqPtr = &((*req)->ibReq);
    }
    this->pimpl_->ctranIb->iput(
        sbuf,
        dbuf,
        len,
        rank,
        mapperRegElem->ibRegElem,
        remoteAccessKey.ibKey,
        notify,
        ibReqPtr);
  }

  return res;
}

ncclResult_t CtranMapper::checkNotify(int rank, bool* notify) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl_->ctranIb) {
    res = this->pimpl_->ctranIb->checkNotify(rank, notify);
  }

  return res;
}

ncclResult_t CtranMapper::waitNotify(int rank) {
  ncclResult_t res = ncclSuccess;

  bool notify = false;
  while (!notify) {
    NCCLCHECKGOTO(this->checkNotify(rank, &notify), res, exit);
  }

exit:
  return res;
}
