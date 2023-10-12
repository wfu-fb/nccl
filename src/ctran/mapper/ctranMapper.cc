// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include "ctranMapper.h"
#include "ctranMapperImpl.h"
#include "comm.h"

ctranMapper::ctranMapper(ncclComm *comm) {
  this->pimpl = std::unique_ptr<impl>(new impl());

  /* regCache */
  this->pimpl->regCache = new class ctranRegCache();

  /* check user preference for backends */
  char *ctranBackendsStr = getenv("NCCL_CTRAN_BACKENDS");
  std::string s;
  if (ctranBackendsStr) {
    s = ctranBackendsStr;
  } else {
    s = "nvl,ib";
  }
  std::string delim = ",";

  while (auto pos = s.find(delim)) {
    std::string b = s.substr(0, pos);
    if (b == "nvl") {
      this->pimpl->backends.push_back(ctranMapperBackend::NVL);
    } else if (b == "ib") {
      this->pimpl->backends.push_back(ctranMapperBackend::IB);
    } else {
      WARN("CTRAN-MAPPER: Unknown backend %s specified", b.c_str());
    }
    s.erase(0, pos + delim.length());
    if (pos == std::string::npos) {
      break;
    }
  }

  /* enable backends that are possible */
  std::vector<enum ctranMapperBackend>::iterator it;

  this->pimpl->ctranIb = nullptr;
  it = std::find(this->pimpl->backends.begin(), this->pimpl->backends.end(),
      ctranMapperBackend::IB);
  if (it != this->pimpl->backends.end()) {
    try {
      this->pimpl->ctranIb = std::unique_ptr<class ctranIb>(new class ctranIb(comm));
    } catch (const std::bad_alloc& e) {
      WARN("CTRAN: IB backend not enabled");
    }
  }

  this->pimpl->ctranNvl = nullptr;
  it = std::find(this->pimpl->backends.begin(), this->pimpl->backends.end(),
      ctranMapperBackend::NVL);
  if (it != this->pimpl->backends.end()) {
    try {
      this->pimpl->ctranNvl = std::unique_ptr<class ctranNvl>(new class ctranNvl(comm));
    } catch (const std::bad_alloc& e) {
      WARN("CTRAN: Nvl backend not enabled");
    }
  }

  for (int i = 0; i < comm->nRanks; i++) {
    /* FIXME: we currently only support NVL for self communication */
    if (i == comm->rank && this->pimpl->ctranNvl != nullptr) {
      this->pimpl->rankBackendMap.push_back(ctranMapperBackend::NVL);
    } else if (this->pimpl->ctranIb != nullptr) {
      this->pimpl->rankBackendMap.push_back(ctranMapperBackend::IB);
    } else {
      this->pimpl->rankBackendMap.push_back(ctranMapperBackend::UNSET);
    }
  }

  CUDACHECKIGNORE(cudaStreamCreateWithFlags(&this->s, cudaStreamNonBlocking));

  this->rank = comm->rank;
  this->commHash = comm->commHash;

  /* Memory pool */
  this->pimpl->memPool = new class ctranMapperMemPool();
  this->pimpl->memPool->regMem(
      [&](const void* buf, std::size_t len, void** hdl) -> ncclResult_t {
          return this->regMem(buf, len, hdl);
      });
}

ctranMapper::~ctranMapper() {
  if (this->pimpl->memPool != nullptr) {
    this->pimpl->memPool->deregMem(
      [&](void* hdl) -> ncclResult_t {
          return this->deregMem(hdl);
      });
  }

  std::vector<void *> v = this->pimpl->regCache->flush();
  if (!v.empty()) {
    WARN("CTRAN-Mapper: found %lu leaked registrations", v.size());
  }
  for (auto hdl : v) {
    NCCLCHECKIGNORE(this->deregMem(hdl));
    WARN("CTRAN-Mapper: leak hdl %p", hdl);
  }
  delete this->pimpl->regCache;

  delete this->pimpl->memPool;

  CUDACHECKIGNORE(cudaStreamDestroy(this->s));

  if (this->pimpl->ctranIb != nullptr) {
    this->pimpl->ctranIb.reset();
  }
  if (this->pimpl->ctranNvl != nullptr) {
    this->pimpl->ctranNvl.reset();
  }
}

ncclResult_t ctranMapper::regMem(const void *buf, std::size_t len, void **hdl) {
  ncclResult_t res = ncclSuccess;

  struct ctranMapperRegElem *regElem = new struct ctranMapperRegElem;

  cudaPointerAttributes attr;
  CUDACHECKGOTO(cudaPointerGetAttributes(&attr, buf), res, exit);
  if (attr.type != cudaMemoryTypeDevice) {
    WARN("CTRAN-MAPPER: buf %p is not a device buffer\n", buf);
    res = ncclSystemError;
    goto exit;
  }

  if (this->pimpl->ctranIb != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranIb->regMem(buf, len, &regElem->ibHdl), res, exit);
  }

  if (this->pimpl->ctranNvl != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranNvl->regMem(buf, len, &regElem->nvlHdl), res, exit);
  }

  NCCLCHECKGOTO(this->pimpl->regCache->insert(buf, len, reinterpret_cast<void *>(regElem), hdl), res, exit);

exit:
  return res;
}

ncclResult_t ctranMapper::deregMem(void *hdl) {
  ncclResult_t res = ncclSuccess;

  struct ctranMapperRegElem *regElem;
  NCCLCHECKGOTO(this->pimpl->regCache->lookup(hdl, (void **) &regElem), res, exit);

  if (this->pimpl->ctranIb != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranIb->deregMem(regElem->ibHdl), res, exit);
  }

  if (this->pimpl->ctranNvl != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranNvl->deregMem(regElem->nvlHdl), res, exit);
  }

  NCCLCHECKGOTO(this->pimpl->regCache->remove(hdl), res, exit);
  delete regElem;

exit:
  return res;
}

ncclResult_t ctranMapper::searchRegHandle(const void *buf, std::size_t len, void **hdl) {
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(this->pimpl->regCache->search(buf, len, hdl), res, exit);

exit:
  return res;
}

ncclResult_t ctranMapper::icopy(void *dbuf, const void *sbuf, std::size_t len, ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  *req = new ctranMapperRequest(this);
  CUDACHECKGOTO(cudaMemcpyAsync(dbuf, sbuf, len, cudaMemcpyDefault, this->s), res, exit);

exit:
  return res;
}

ncclResult_t ctranMapper::progress(void) {
  if (this->pimpl->ctranIb != nullptr) {
    NCCLCHECK(this->pimpl->ctranIb->progress());
  }
  if (this->pimpl->ctranNvl != nullptr) {
    NCCLCHECK(this->pimpl->ctranNvl->progress());
  }

  return ncclSuccess;
}

ncclResult_t ctranMapper::getTmpBuf(void** addr, std::size_t len, void **hdl) {
    this->tmpBufLock.lock();
    *hdl = nullptr;
    std::size_t bufLen;
    NCCLCHECK(this->pimpl->memPool->getBuf(len, addr, hdl, &bufLen));
    if (*hdl == nullptr) {
      NCCLCHECK(this->regMem(*addr, bufLen, hdl));
    }
    this->tmpBufLock.unlock();
    return ncclSuccess;
}

ncclResult_t ctranMapper::releaseTmpBuf(void* addr, void *hdl) {
    this->tmpBufLock.lock();
    NCCLCHECK(this->pimpl->memPool->release(addr, hdl));
    this->tmpBufLock.unlock();

    return ncclSuccess;
}

ncclResult_t ctranMapper::isendCtrl(void *buf, void *hdl, int rank, ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    struct ctranMapperRegElem *regElem;
    NCCLCHECKGOTO(this->pimpl->regCache->lookup(hdl, (void **) &regElem), res, exit);

    if (req == nullptr) {
      return this->pimpl->ctranIb->isendCtrl(buf, regElem->ibHdl, rank, nullptr);
    } else {
      *req = new ctranMapperRequest(this);
      NCCLCHECKGOTO(this->pimpl->ctranIb->isendCtrl(buf, regElem->ibHdl, rank, &((*req)->ibReq)), res, exit);
    }
  }

exit:
  return ncclSuccess;
}

ncclResult_t ctranMapper::irecvCtrl(void **buf, struct ctranMapperRemoteAccessKey *key, int rank,
    ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    if (req == nullptr) {
      return this->pimpl->ctranIb->irecvCtrl(buf, &key->ibKey, rank, nullptr);
    } else {
      *req = new ctranMapperRequest(this);
      NCCLCHECKGOTO(this->pimpl->ctranIb->irecvCtrl(buf, &key->ibKey, rank, &((*req)->ibReq)), res, exit);
    }
  }

exit:
  return ncclSuccess;
}

ncclResult_t ctranMapper::iput(const void *sbuf, void *dbuf, std::size_t len, int rank, void *shdl,
    struct ctranMapperRemoteAccessKey remoteAccessKey, bool notify, ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    struct ctranMapperRegElem *regElem;
    NCCLCHECKGOTO(this->pimpl->regCache->lookup(shdl, (void **) &regElem), res, exit);

    if (req == nullptr) {
      NCCLCHECKGOTO(this->pimpl->ctranIb->iput(sbuf, dbuf, len, rank, regElem->ibHdl, remoteAccessKey.ibKey,
            notify, nullptr), res, exit);
    } else {
      *req = new ctranMapperRequest(this);
      NCCLCHECKGOTO(this->pimpl->ctranIb->iput(sbuf, dbuf, len, rank, regElem->ibHdl, remoteAccessKey.ibKey,
            notify, &((*req)->ibReq)), res, exit);
    }
  }

exit:
  return res;
}

ncclResult_t ctranMapper::checkNotify(int rank, bool *notify) {
  ncclResult_t res = ncclSuccess;

  if (this->pimpl->ctranIb != nullptr) {
    NCCLCHECKGOTO(this->pimpl->ctranIb->checkNotify(rank, notify), res, exit);
  }

exit:
  return ncclSuccess;
}

ncclResult_t ctranMapper::waitNotify(int rank) {
  ncclResult_t res = ncclSuccess;

  bool notify = false;
  while (notify == false) {
    NCCLCHECKGOTO(this->checkNotify(rank, &notify), res, exit);
  }

exit:
  return ncclSuccess;
}

