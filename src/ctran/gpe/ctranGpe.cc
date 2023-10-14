// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranGpe.h"
#include "ctranGpeImpl.h"
#include "checks.h"
#include <iostream>

ctranGpe::ctranGpe(int cudaDev) {
  this->pimpl = std::unique_ptr<impl>(new impl());
  this->pimpl->t = std::thread{ctranGpe::impl::gpeThreadFn, this->pimpl.get(), cudaDev};
}

ctranGpe::~ctranGpe() {
  std::vector<std::unique_ptr<struct collOp>> empty;
  this->pimpl->submit(ctranGpeCmd::typeEnum::TERMINATE, std::move(empty), nullptr, nullptr);
  this->pimpl->t.join();
}

ncclResult_t ctranGpe::submit(std::vector<std::unique_ptr<struct collOp>> opGroup,
    collOpFunc func, const void *ncclKernel) {
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(this->pimpl->submit(ctranGpeCmd::typeEnum::GRAPH_ENQUEUE, std::move(opGroup),
        func, ncclKernel), res, exit);

exit:
  return res;
}
