// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CtranMapper.h"
#include "debug.h"

CtranMapperRequest::~CtranMapperRequest() {
  if (this->ibReq != nullptr) {
    delete this->ibReq;
  }
}

ncclResult_t CtranMapperRequest::test(bool* isComplete) {
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(this->mapper_->progress(), res, exit);

  *isComplete = false;
  if (this->ibReq != nullptr) {
    *isComplete = this->ibReq->isComplete();
  } else {
    auto cudaErr = cudaStreamQuery(this->mapper_->internalStream);
    if (cudaErr == cudaSuccess) {
      *isComplete = true;
    } else if (cudaErr != cudaErrorNotReady) {
      WARN(
          "CTRAN: cudaStreamQuery returned error '%s'\n",
          cudaGetErrorString(cudaErr));
      res = ncclSystemError;
      goto exit;
    }
  }

  if (*isComplete) {
    this->state_ = CtranMapperRequest::COMPLETE;
  }

exit:
  return res;
}

ncclResult_t CtranMapperRequest::wait(void) {
  ncclResult_t res = ncclSuccess;
  bool isComplete = false;

  while (isComplete == false) {
    NCCLCHECKGOTO(this->test(&isComplete), res, exit);
  }

exit:
  return res;
}
