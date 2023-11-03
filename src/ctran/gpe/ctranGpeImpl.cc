// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranGpe.h"
#include "ctranGpeImpl.h"
#include "ctranGpeKernel.h"
#include "checks.h"
#include <iostream>

ctranGpe::impl::impl() {
  CUDACHECKIGNORE(cudaHostAlloc(&this->kernelFlag, sizeof(int), cudaHostAllocDefault));
  *(this->kernelFlag) = 0;
}

ctranGpe::impl::~impl() {
  CUDACHECKIGNORE(cudaFreeHost(this->kernelFlag));
}

ncclResult_t ctranGpe::impl::submit(ctranGpeCmd::typeEnum type,
    std::vector<std::unique_ptr<struct collOp>> opGroup, collOpFunc func, const void *ncclKernel) {
  ncclResult_t res = ncclSuccess;

  struct ctranGpeCmd *cmd = new struct ctranGpeCmd;
  cmd->type = type;

  if (type == ctranGpeCmd::typeEnum::GRAPH_ENQUEUE) {
    /* post copy to user stream */
    if (opGroup.front()->type == collOp::opType::COPY) {
      CUDACHECKGOTO(
          cudaMemcpyAsync(
              opGroup.front()->copy.dst,
              opGroup.front()->copy.src,
              opGroup.front()->copy.nbytes,
              cudaMemcpyDefault,
              opGroup.front()->stream),
          res,
          exit);
      opGroup.erase(opGroup.begin());
    }

    cudaStream_t stream = opGroup.front()->stream;
    cmd->coll.opGroup = std::move(opGroup);
    cmd->coll.func = func;

    /* Enqueue the kernel.  It will not start till all other
     * operations on this stream have completed. */
    dim3 grid = { 1, 1, 1 };
    dim3 blocks = { 1, 1, 1 };
    void *args[] = { &this->kernelFlag };
    CUDACHECKGOTO(cudaLaunchKernel(ncclKernel, grid, blocks, args, 0, stream), res, exit);
  }

  this->m.lock();
  cmdQueue.push(cmd);
  this->m.unlock();
  c.notify_one();

exit:
  return res;
}

void ctranGpe::impl::gpeThreadFn(ctranGpe::impl *pimpl, int cudaDev) {
  CUDACHECKIGNORE(cudaSetDevice(cudaDev));

  while (1) {
    ctranGpeCmd *cmd;

    {
      std::unique_lock<std::mutex> lk(pimpl->m);
      pimpl->c.wait(lk, [&] { return !pimpl->cmdQueue.empty(); } );

      cmd = pimpl->cmdQueue.front();
      pimpl->cmdQueue.pop();
    }

    if (cmd->type == ctranGpeCmd::typeEnum::TERMINATE) {
      goto exit;
    }

    /* wait for the kernel to launch */
    volatile int *flag_d = pimpl->kernelFlag;
    while (*flag_d != KERNEL_STARTED);

    /* run collective */
    NCCLCHECKIGNORE(cmd->coll.func(std::move(cmd->coll.opGroup)));

    /* stop kernel */
    *flag_d = KERNEL_TERMINATE;

    delete cmd;
  }

exit:
  return;
}

