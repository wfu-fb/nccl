// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CtranGpe.h"
#include <iostream>
#include "CtranGpeImpl.h"
#include "checks.h"
#include "comm.h"

OpElem::OpElem(enum opType type, ncclComm_t comm) : type(type), comm(comm) {
  if (type == ALLTOALLV) {
    new (&this->alltoallv.sendcounts) std::vector<size_t>;
    this->alltoallv.sendcounts.resize(comm->nRanks);
    new (&this->alltoallv.sdispls) std::vector<size_t>;
    this->alltoallv.sdispls.resize(comm->nRanks);
    new (&this->alltoallv.recvcounts) std::vector<size_t>;
    this->alltoallv.recvcounts.resize(comm->nRanks);
    new (&this->alltoallv.rdispls) std::vector<size_t>;
    this->alltoallv.rdispls.resize(comm->nRanks);
  }
}

OpElem::OpElem(enum opType type, cudaStream_t stream, ncclComm_t comm)
    : type(type), stream(stream), comm(comm) {}

OpElem::~OpElem() {
  if (type == ALLTOALLV) {
    this->alltoallv.sendcounts.~vector();
    this->alltoallv.sdispls.~vector();
    this->alltoallv.recvcounts.~vector();
    this->alltoallv.rdispls.~vector();
  }
}

CtranGpe::CtranGpe(int cudaDev) {
  this->pimpl = std::unique_ptr<Impl>(new Impl());
  this->pimpl->t =
      std::thread{CtranGpe::Impl::gpeThreadFn, this->pimpl.get(), cudaDev};
}

CtranGpe::~CtranGpe() {
  this->pimpl->terminate();
  this->pimpl->t.join();
}

ncclResult_t CtranGpe::submit(
    std::vector<std::unique_ptr<struct OpElem>> opGroup,
    opFunc func,
    KernelConfig& kernelConfig,
    const void* ncclKernel) {
  return this->pimpl->submit(
      CtranGpeCmd::TypeEnum::GRAPH_ENQUEUE,
      std::move(opGroup),
      func,
      kernelConfig,
      ncclKernel);
}

ncclResult_t CtranGpe::allocKernelP2pElems(
    size_t numElems,
    int ngroups,
    KernelP2pElem** elemsList) {
  // wait outstanding kernels to release inused elems.
  while (numElems > this->pimpl->kernelP2pElemPool->size()) {
    this->pimpl->kernelP2pElemPool->reclaim();
  }

  // pop free elements and put into C style list for kernel to use.
  if (numElems > 0) {
    *elemsList = this->pimpl->kernelP2pElemPool->pop(ngroups);
    if (!*elemsList) {
      return ncclInternalError;
    }
  }
  auto elem = *elemsList;
  for (int i = 1; i < numElems; i++) {
    elem->next = this->pimpl->kernelP2pElemPool->pop(ngroups);
    if (!elem->next) {
      return ncclInternalError;
    }
    elem = elem->next;
  }
  return ncclSuccess;
}
