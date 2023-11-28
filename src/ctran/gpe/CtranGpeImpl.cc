// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CtranGpeImpl.h"
#include <iostream>
#include "CtranGpe.h"
#include "CtranGpeKernel.h"
#include "checks.h"

CtranGpe::Impl::Impl() {
  CUDACHECKIGNORE(
      cudaHostAlloc(&this->kernelFlag, sizeof(int), cudaHostAllocDefault));
  *(this->kernelFlag) = UNSET;
}

CtranGpe::Impl::~Impl() {
  CUDACHECKIGNORE(cudaFreeHost(this->kernelFlag));
}

ncclResult_t CtranGpe::Impl::submit(
    CtranGpeCmd::TypeEnum type,
    std::vector<std::unique_ptr<struct OpElem>> opGroup,
    opFunc func,
    const void* ncclKernel) {
  ncclResult_t res = ncclSuccess;

  struct CtranGpeCmd* cmd = new struct CtranGpeCmd;
  cmd->type = type;

  if (type == CtranGpeCmd::TypeEnum::GRAPH_ENQUEUE) {
    cudaStream_t stream = opGroup.front()->stream;
    cmd->coll.opGroup = std::move(opGroup);
    cmd->coll.func = func;

    /* Enqueue the kernel.  It will not start till all other
     * operations on this stream have completed. */
    dim3 grid = {1, 1, 1};
    dim3 blocks = {1, 1, 1};
    void* args[] = {&this->kernelFlag};
    CUDACHECKGOTO(
        cudaLaunchKernel(ncclKernel, grid, blocks, args, 0, stream), res, exit);
  }

  this->m.lock();
  cmdQueue.push(cmd);
  this->m.unlock();
  c.notify_one();

exit:
  return res;
}

void CtranGpe::Impl::gpeThreadFn(CtranGpe::Impl* pimpl, int cudaDev) {
  CUDACHECKIGNORE(cudaSetDevice(cudaDev));

  while (1) {
    CtranGpeCmd* cmd;

    {
      std::unique_lock<std::mutex> lk(pimpl->m);
      pimpl->c.wait(lk, [&] { return !pimpl->cmdQueue.empty(); });

      cmd = pimpl->cmdQueue.front();
      pimpl->cmdQueue.pop();
    }

    if (cmd->type == CtranGpeCmd::TypeEnum::TERMINATE) {
      goto exit;
    }

    /* wait for the kernel to launch */
    volatile int* flag_d = pimpl->kernelFlag;
    while (*flag_d != KERNEL_STARTED)
      ;

    /* run collective */
    NCCLCHECKIGNORE(cmd->coll.func(std::move(cmd->coll.opGroup)));

    /* stop kernel */
    *flag_d = KERNEL_TERMINATE;

    delete cmd;
  }

exit:
  return;
}
