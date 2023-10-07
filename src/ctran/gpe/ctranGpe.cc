// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranGpe.h"
#include "ctranGpeImpl.h"
#include <iostream>

ctranGpe::ctranGpe(int cudaDev) {
  this->pimpl = std::unique_ptr<impl>(new impl());
  this->pimpl->t = std::thread{ctranGpe::impl::gpeThreadFn, this->pimpl.get(), cudaDev};
}

ctranGpe::~ctranGpe() {
  this->pimpl->enqueue(ctranGpeCmd::typeEnum::TERMINATE, nullptr, nullptr);
  this->pimpl->t.join();
}

ncclResult_t ctranGpe::submit(std::unique_ptr<ctranGraph> graph, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(this->pimpl->enqueue(ctranGpeCmd::typeEnum::GRAPH_ENQUEUE, std::move(graph), stream), res, exit);

exit:
  return res;
}
