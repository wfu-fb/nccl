// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "nccl.h"
#include "argcheck.h"
#include "comm.h"

NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 1);

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  if (ncclParamLocalRegister()) {
    NCCLCHECK(PtrCheck(comm, "ncclCommRegister", "comm"));

    NCCLCHECK(comm->ctranMapper->regMem(buff, size, handle));
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  if (ncclParamLocalRegister()) {
    NCCLCHECK(PtrCheck(comm, "ncclCommRegister", "comm"));

    NCCLCHECK(comm->ctranMapper->deregMem(handle));
  }
  return ncclSuccess;
}
