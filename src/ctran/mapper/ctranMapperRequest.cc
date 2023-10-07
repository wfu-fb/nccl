// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranMapper.h"
#include "ctranMapperRequestImpl.h"
#include "debug.h"

ctranMapperRequest::ctranMapperRequest() {
  this->pimpl = std::unique_ptr<impl>(new impl());
  this->pimpl->state = ctranMapperRequest::impl::CTRAN_REQUEST_STATE_INCOMPLETE;
  this->ibReq = nullptr;
  this->nvlReq = nullptr;
}

ctranMapperRequest::~ctranMapperRequest() {
  if (this->ibReq != nullptr) {
    delete this->ibReq;
  }
  if (this->nvlReq != nullptr) {
    delete this->nvlReq;
  }
}

ncclResult_t ctranMapperRequest::test(bool *isComplete) {
  ncclResult_t res = ncclSuccess;

  *isComplete = false;
  if (this->ibReq != nullptr) {
    NCCLCHECKGOTO(this->ibReq->test(isComplete), res, exit);
  } else if (this->nvlReq != nullptr) {
    NCCLCHECKGOTO(this->nvlReq->test(isComplete), res, exit);
  } else {
    auto cudaErr = cudaEventQuery(this->e);
    if (cudaErr == cudaSuccess) {
      *isComplete = true;
      CUDACHECKGOTO(cudaEventDestroy(this->e), res, exit);
    } else if (cudaErr != cudaErrorNotReady) {
      WARN("CTRAN-NVL: cudaEventQuery returned error '%s'\n", cudaGetErrorString(cudaErr));
      res = ncclSystemError;
      goto exit;
    }
  }

  if (*isComplete) {
    this->pimpl->state = ctranMapperRequest::impl::CTRAN_REQUEST_STATE_COMPLETE;
  }

exit:
  return res;
}

uint64_t ctranMapperRequest::getWaitTime() {
  if (this->ibReq != nullptr) {
    return this->ibReq->getWaitTime();
  }
  if (this->nvlReq != nullptr) {
    return this->nvlReq->getWaitTime();
  }

  return 0;
}

uint64_t ctranMapperRequest::getCommTime() {
  if (this->ibReq != nullptr) {
    return this->ibReq->getCommTime();
  }
  if (this->nvlReq != nullptr) {
    return this->nvlReq->getCommTime();
  }

  return 0;
}
