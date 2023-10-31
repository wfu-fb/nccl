// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_AVL_TREE_ELEM_H_
#define CTRAN_AVL_TREE_ELEM_H_

#include <vector>
#include "nccl.h"

class ctranAvlTree::TreeElem {
  public:
    TreeElem(uintptr_t addr, std::size_t len, void *val);
    ~TreeElem(void);
    void updateHeight(void);

    TreeElem *insert(uintptr_t addr, std::size_t len, void *val, TreeElem **hdl);
    TreeElem *remove(void);
    void print(int level);

    uintptr_t addr;
    std::size_t len;
    void *val;
    TreeElem *left;
    TreeElem *right;

  private:
    TreeElem *leftRotate(void);
    TreeElem *rightRotate(void);
    TreeElem *balance(void);
    uint32_t height;
};

#endif
