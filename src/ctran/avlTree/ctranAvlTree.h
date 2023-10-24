// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_AVL_TREE_H_
#define CTRAN_AVL_TREE_H_

#include <cstdint>
#include <vector>
#include <memory>
#include "nccl.h"

class ctranAvlTree {
public:
  ctranAvlTree();
  ~ctranAvlTree();

  ncclResult_t insert(const void *addr, std::size_t len, void *val, void **hdl);
  ncclResult_t remove(void *hdl);
  ncclResult_t search(const void *addr, std::size_t len, void **hdl);
  ncclResult_t lookup(void *hdl, void **val);
  std::vector<void *> getAllElems();

private:
  class TreeVector;
  class TreeVector *root;
};

#endif
