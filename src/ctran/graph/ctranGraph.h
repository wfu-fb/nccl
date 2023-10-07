// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GRAPH_H_
#define CTRAN_GRAPH_H_

#include <vector>
#include <string>
#include <functional>
#include "nccl.h"
#include "ctranMapper.h"

class ctranGraph {
public:
  ctranGraph(ctranMapper *mapper, std::string name);
  ~ctranGraph();
  ncclResult_t isend(const void *buf, std::size_t len, int rank, void *hdl, std::vector<int> deps, int *opHandle);
  ncclResult_t irecv(void *buf, std::size_t len, int rank, void *hdl, std::vector<int> deps, int *opHandle);
  ncclResult_t icopy(void *dbuf, const void *sbuf, std::size_t len, std::vector<int> deps, int *opHandle);
  ncclResult_t test(bool *isComplete);
  ncclResult_t registerCB(std::function<void(void)> hook);

  const void *ncclKernel;

private:
  class impl;
  std::unique_ptr<impl> pimpl;
};

#endif
