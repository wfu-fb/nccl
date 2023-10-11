// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_IMPL_H_
#define CTRAN_GPE_IMPL_H_

#include "ctranGpe.h"
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>

class ctranGpeCmd {
  public:
    ctranGpeCmd() = default;
    ~ctranGpeCmd() = default;

    enum typeEnum {
      GRAPH_ENQUEUE,
      TERMINATE,
    } type;

    struct {
      std::unique_ptr<struct collOp> op;
      cudaStream_t stream;
    } coll;
};

class ctranGpe::impl {
  public:
    impl();
    ~impl();

    ncclResult_t enqueue(ctranGpeCmd::typeEnum type, std::unique_ptr<struct collOp> op,
        cudaStream_t stream);
    static void gpeThreadFn(class ctranGpe::impl *pimpl, int cudaDev);

    std::thread t;
    std::queue<ctranGpeCmd *> cmdQueue;
    std::queue<ctranGpeCmd *> internalCmdQueue;
    std::mutex m;
    std::condition_variable c;
    int *kernelFlag;
};

#endif
