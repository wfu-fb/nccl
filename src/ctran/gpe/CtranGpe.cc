// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CtranGpe.h"
#include <iostream>
#include "CtranGpeImpl.h"
#include "checks.h"

CtranGpe::CtranGpe(int cudaDev) {
  this->pimpl = std::unique_ptr<Impl>(new Impl());
  this->pimpl->t =
      std::thread{CtranGpe::Impl::gpeThreadFn, this->pimpl.get(), cudaDev};
}

CtranGpe::~CtranGpe() {
  std::vector<std::unique_ptr<struct OpElem>> empty;
  this->pimpl->submit(
      CtranGpeCmd::TypeEnum::TERMINATE, std::move(empty), nullptr, nullptr);
  this->pimpl->t.join();
}

ncclResult_t CtranGpe::submit(
    std::vector<std::unique_ptr<struct OpElem>> opGroup,
    opFunc func,
    const void* ncclKernel) {
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(
      this->pimpl->submit(
          CtranGpeCmd::TypeEnum::GRAPH_ENQUEUE,
          std::move(opGroup),
          func,
          ncclKernel),
      res,
      exit);

exit:
  return res;
}
