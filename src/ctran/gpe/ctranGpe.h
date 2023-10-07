// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GPE_H_
#define CTRAN_GPE_H_

#include "ctranGraph.h"

class ctranGpe {
  public:
    ctranGpe(int cudaDev);
    ~ctranGpe();
    ncclResult_t submit(std::unique_ptr<ctranGraph> graph, cudaStream_t stream);

  private:
    class impl;
    std::unique_ptr<impl> pimpl;
};

__global__ void ncclKernelAllGatherCTD(int *flag);
__global__ void ncclKernelAllGatherCTR(int *flag);

#endif
