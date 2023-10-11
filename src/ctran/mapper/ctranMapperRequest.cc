// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranMapper.h"
#include "debug.h"

ctranMapperRequest::ctranMapperRequest(ctranMapper *mapper, cudaEvent_t e) {
  this->mapper = mapper;
  this->state = ctranMapperRequest::INCOMPLETE;
  this->ibReq = nullptr;
  this->nvlReq = nullptr;
  this->e = e;
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

  NCCLCHECKGOTO(this->mapper->progress(), res, exit);

  *isComplete = false;
  if (this->ibReq != nullptr) {
    NCCLCHECKGOTO(this->ibReq->test(isComplete), res, exit);
  } else if (this->nvlReq != nullptr) {
    NCCLCHECKGOTO(this->nvlReq->test(isComplete), res, exit);
  } else {
    auto cudaErr = cudaEventQuery(this->e);
    if (cudaErr == cudaSuccess) {
      *isComplete = true;
    } else if (cudaErr != cudaErrorNotReady) {
      WARN("CTRAN-NVL: cudaEventQuery returned error '%s'\n", cudaGetErrorString(cudaErr));
      res = ncclSystemError;
      goto exit;
    }
  }

  if (*isComplete) {
    this->state = ctranMapperRequest::COMPLETE;
  }

exit:
  return res;
}

ncclResult_t ctranMapperRequest::wait(void) {
  ncclResult_t res = ncclSuccess;
  bool isComplete = false;

  while (isComplete == false) {
    NCCLCHECKGOTO(this->test(&isComplete), res, exit);
  }

exit:
  return res;
}
