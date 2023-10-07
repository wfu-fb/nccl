// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_GRAPH_IMPL_H_
#define CTRAN_GRAPH_IMPL_H_

#include <vector>
#include "nccl.h"
#include "ctranMapper.h"

struct ctranGraphElem {
  enum elemType {
    ISEND,
    IRECV,
    ICOPY,
  } type;
  int opHandle;

  /* to be scheduled after upstream dependencies are complete */
  std::vector<struct ctranGraphElem *> upstreamDeps;
  /* notify downstream dependencies when done */
  std::vector<struct ctranGraphElem *> downstreamDeps;

  union {
    struct {
      const void *buf;
      std::size_t len;
      int rank;
      void *hdl;
    } isend;
    struct {
      void *buf;
      std::size_t len;
      int rank;
      void *hdl;
    } irecv;
    struct {
      const void *sbuf;
      void *dbuf;
      std::size_t len;
    } icopy;
  } u;

  ctranMapperRequest *req;
};

struct timestamp {
  uint64_t waitTime;
  uint64_t commTime;
  int peer;
  size_t len;
};

class ctranGraph::impl {
public:
  impl() = default;
  ~impl() = default;

  ctranMapper *mapper;
  std::vector<struct ctranGraphElem *> allOps;
  std::vector<struct ctranGraphElem *> readyOps;
  std::vector<struct ctranGraphElem *> postedOps;
  int opHandleCounter;

  ncclResult_t processGraph();

  std::vector<struct timestamp> timestamps;
  std::string name;
};

#endif
