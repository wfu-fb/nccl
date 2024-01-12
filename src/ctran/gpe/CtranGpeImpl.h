// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_IMPL_H_
#define CTRAN_GPE_IMPL_H_

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include "CtranGpe.h"

class CtranGpeCmd {
 public:
  CtranGpeCmd() = default;
  ~CtranGpeCmd() = default;

  enum TypeEnum {
    GRAPH_ENQUEUE,
    TERMINATE,
  } type;

  struct {
    std::vector<std::unique_ptr<struct OpElem>> opGroup;
    opFunc func;
  } coll;
};

class CtranGpe::Impl {
 public:
  Impl();
  ~Impl();

  ncclResult_t submit(
      CtranGpeCmd::TypeEnum type,
      std::vector<std::unique_ptr<struct OpElem>> opGroup,
      opFunc func,
      KernelConfig& kernelConfig,
      const void* ncclKernel);

  ncclResult_t terminate();

  static void gpeThreadFn(class CtranGpe::Impl* pimpl, int cudaDev);

  std::thread t;
  std::queue<CtranGpeCmd*> cmdQueue;
  std::mutex m;
  std::condition_variable c;
  int* kernelFlag{nullptr};
};

#endif
