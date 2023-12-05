// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_COMM_H_
#define CTRAN_COMM_H_

#include "CtranGpe.h"
#include "CtranMapper.h"
#include "nccl.h"

struct ncclComm;

class Ctran {
 public:
  Ctran(ncclComm* comm);
  ~Ctran() = default;

 public:
  std::unique_ptr<CtranMapper> mapper{nullptr};
  std::unique_ptr<CtranGpe> gpe{nullptr};
};

ncclResult_t ctranInit(ncclComm* comm);
bool ctranInitialized(ncclComm* comm);
ncclResult_t ctranDestroy(ncclComm* comm);

#endif // CTRAN_COMM_H_
