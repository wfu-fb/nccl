// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_AVL_TREE_H_
#define CTRAN_AVL_TREE_H_

#include <cstdint>
#include <vector>
#include <memory>

class ctranAvlTree {
public:
  ctranAvlTree();
  ~ctranAvlTree();

  void insert(const void *addr, std::size_t len, void *val, void **hdl);
  void remove(void *hdl);
  void search(const void *addr, std::size_t len, void **hdl);
  void lookup(void *hdl, void **val);
  std::vector<void *> getAllElems();

private:
  class TreeElem;
  class TreeElem *root;
  std::vector<TreeElem *> list;
};

#endif
