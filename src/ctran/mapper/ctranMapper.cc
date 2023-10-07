// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>
#include "ctranMapper.h"
#include "ctranMapperImpl.h"
#include "comm.h"

ctranMapper::ctranMapper(ncclComm *comm, ncclComm *parent, int *parentRanks) {
  this->pimpl = std::unique_ptr<impl>(new impl());

  /* rank mapping */
  if (parent == nullptr) {
    for (int i = 0; i < comm->nRanks; i++) {
      this->pimpl->rankMap.push_back(i);
    }
  } else {
    for (int i = 0; i < comm->nRanks; i++) {
      this->pimpl->rankMap.push_back(parent->ctranMapper->pimpl->rankMap[parentRanks[i]]);
    }
  }

  /* regCache */
  this->pimpl->regCache = new class ctranRegCache();

  /* get unique Id */
  if (parent == nullptr) {
    this->pimpl->sharedMapper = std::make_shared<class ctranMapperShared>();
  } else {
    this->pimpl->sharedMapper = parent->ctranMapper->pimpl->sharedMapper;
  }
  NCCLCHECKIGNORE(this->pimpl->sharedMapper->getUniqueId(comm, &this->pimpl->uniqueId));

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
  if (parent == nullptr) {
    std::vector<enum ctranMapperBackend>::iterator it;

    this->pimpl->ctranIb = nullptr;
    this->pimpl->ctranNvl = nullptr;

    it = std::find(this->pimpl->backends.begin(), this->pimpl->backends.end(),
        ctranMapperBackend::IB);
    if (it != this->pimpl->backends.end()) {
      try {
        this->pimpl->ctranIb = std::shared_ptr<class ctranIb>(new class ctranIb(comm));
      } catch (const std::bad_alloc& e) {
        WARN("CTRAN: IB backend not enabled");
      }
    }

    it = std::find(this->pimpl->backends.begin(), this->pimpl->backends.end(),
        ctranMapperBackend::NVL);
    if (it != this->pimpl->backends.end()) {
      try {
        this->pimpl->ctranNvl = std::shared_ptr<class ctranNvl>(new class ctranNvl(comm));
      } catch (const std::bad_alloc& e) {
        WARN("CTRAN: Nvl backend not enabled");
      }
    }
  } else {
    this->pimpl->ctranIb = parent->ctranMapper->pimpl->ctranIb;
    this->pimpl->ctranNvl = parent->ctranMapper->pimpl->ctranNvl;
  }

  for (int i = 0; i < comm->nRanks; i++) {
    /* FIXME: we currently only support NVL for self communication */
    if (i == comm->rank && this->pimpl->ctranNvl != nullptr) {
      this->pimpl->rankBackendMap.push_back(CTRAN_BACKEND_NVL);
    } else if (this->pimpl->ctranIb != nullptr) {
      this->pimpl->rankBackendMap.push_back(CTRAN_BACKEND_IB);
    } else {
      this->pimpl->rankBackendMap.push_back(CTRAN_BACKEND_UNSET);
    }
  }

  CUDACHECKIGNORE(cudaStreamCreateWithFlags(&this->pimpl->s, cudaStreamNonBlocking));

  this->rank = comm->rank;
  this->commHash = comm->commHash;
  this->collId = 0;

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

  CUDACHECKIGNORE(cudaStreamDestroy(this->pimpl->s));

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

ncclResult_t ctranMapper::isend(const void *buf, std::size_t len, int peerRank, void *hdl, ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  auto r = new ctranMapperRequest();
  struct ctranMapperRegElem *regElem;
  NCCLCHECKGOTO(this->pimpl->regCache->lookup(hdl, (void **) &regElem), res, exit);

  switch (this->pimpl->rankBackendMap[peerRank]) {
    case CTRAN_BACKEND_IB:
      {
        NCCLCHECKGOTO(this->pimpl->ctranIb->isend(buf, len, this->pimpl->rankMap[peerRank],
              regElem->ibHdl, this->pimpl->uniqueId, &r->ibReq), res, exit);
      }
      break;

    case CTRAN_BACKEND_NVL:
      {
        NCCLCHECKGOTO(this->pimpl->ctranNvl->isend(buf, len, this->pimpl->rankMap[peerRank],
              regElem->nvlHdl, this->pimpl->uniqueId, &r->nvlReq), res, exit);
      }
      break;

    default:
      res = ncclSystemError;
      goto exit;
  }

  *req = r;

exit:
  return res;
}

ncclResult_t ctranMapper::irecv(void *buf, std::size_t len, int peerRank, void *hdl, ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  auto r = new ctranMapperRequest();
  struct ctranMapperRegElem *regElem;
  NCCLCHECKGOTO(this->pimpl->regCache->lookup(hdl, (void **) &regElem), res, exit);

  switch (this->pimpl->rankBackendMap[peerRank]) {
    case CTRAN_BACKEND_IB:
      {
        NCCLCHECKGOTO(this->pimpl->ctranIb->irecv(buf, len, this->pimpl->rankMap[peerRank],
              regElem->ibHdl, this->pimpl->uniqueId, &r->ibReq), res, exit);
      }
      break;

    case CTRAN_BACKEND_NVL:
      {
        NCCLCHECKGOTO(this->pimpl->ctranNvl->irecv(buf, len, this->pimpl->rankMap[peerRank],
              regElem->nvlHdl, this->pimpl->uniqueId, &r->nvlReq), res, exit);
      }
      break;

    default:
      res = ncclSystemError;
      goto exit;
  }

  *req = r;

exit:
  return res;
}

ncclResult_t ctranMapper::icopy(void *dbuf, const void *sbuf, std::size_t len, ctranMapperRequest **req) {
  ncclResult_t res = ncclSuccess;

  auto r = new ctranMapperRequest();

  CUDACHECKGOTO(cudaMemcpyAsync(dbuf, sbuf, len, cudaMemcpyDefault, this->pimpl->s), res, exit);
  CUDACHECKGOTO(cudaEventCreate(&r->e), res, exit);
  CUDACHECKGOTO(cudaEventRecord(r->e, this->pimpl->s), res, exit);

  *req = r;

exit:
  return res;
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
