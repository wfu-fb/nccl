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
    KernelConfig& kernelConfig,
    const void* ncclKernel) {
  // Set first kernel argument as kernelFlag if GPE op is not empty.
  // Check it before passing opGroup to cmd
  std::vector<void*> kernelArgs;
  void* kernelFlag = opGroup.size() ? this->kernelFlag : nullptr;
  kernelArgs.push_back(&kernelFlag);

  // Enqueue op to gpeThread if any op is appended
  if (!opGroup.empty()) {
    struct CtranGpeCmd* cmd = new struct CtranGpeCmd;
    cmd->type = type;

    if (type == CtranGpeCmd::TypeEnum::GRAPH_ENQUEUE) {
      cmd->coll.opGroup = std::move(opGroup);
      cmd->coll.func = func;
    }

    this->m.lock();
    cmdQueue.push(cmd);
    this->m.unlock();
    c.notify_one();
  }

  // Enqueue the kernel with arguments.  It will not start till all other
  // operations on this stream have completed
  dim3 grid = {kernelConfig.numBlocks, 1, 1};
  dim3 blocks = {kernelConfig.numThreads, 1, 1};

  // Specify collective arguments
  kernelArgs.push_back((void*)&kernelConfig.args.devState_d);
  kernelArgs.push_back((void*)&kernelConfig.args.collective);

  CUDACHECK(cudaLaunchKernel(
      ncclKernel, grid, blocks, kernelArgs.data(), 0, kernelConfig.stream));

  return ncclSuccess;
}

ncclResult_t CtranGpe::Impl::terminate() {
  struct CtranGpeCmd* cmd = new struct CtranGpeCmd;
  cmd->type = CtranGpeCmd::TypeEnum::TERMINATE;

  this->m.lock();
  cmdQueue.push(cmd);
  this->m.unlock();
  c.notify_one();

  return ncclSuccess;
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
