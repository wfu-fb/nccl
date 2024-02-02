// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_IMPL_H_
#define CTRAN_GPE_IMPL_H_

#include <condition_variable>
#include <list>
#include <mutex>
#include <queue>
#include <stack>
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

/**
 * Pool of KernelP2pElem objects allocated from cudaHostAlloc. It is NOT
 * thread-safe, as one pool for every GPE and both pop and reclaim operations
 * are expected to be called from main thread before submitting a new command to
 * the GPE.
 */
class KernelP2pElemPool {
 public:
  KernelP2pElemPool(size_t capacity);
  ~KernelP2pElemPool();

  // Pop a KernelP2pElem from the free pool; enqueue to the in-use queue for
  // later reclaimant
  // Input arguments:
  //   - ngroups: number of thread block groups to use each p2pElem object; used
  //              to set inuse flag
  KernelP2pElem* pop(int ngroups);

  // Reclaim any unused KernelP2pElem objects back to the free pool.
  void reclaim();

  // Return the number of KernelP2pElem objects in the free pool.
  size_t size();

 private:
  void resetWorkElem(KernelP2pElem* workElem);

  std::stack<KernelP2pElem*> freeWorkElems_;
  std::list<KernelP2pElem*> inuseWorkElems_;
  size_t capacity_{0};
  void* memPtr_{nullptr};
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
  std::unique_ptr<KernelP2pElemPool> kernelP2pElemPool;
};

#endif
