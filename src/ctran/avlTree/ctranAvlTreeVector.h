// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_AVL_TREE_VECTOR_H_
#define CTRAN_AVL_TREE_VECTOR_H_

#include <vector>
#include "nccl.h"

class ctranAvlTree::TreeVector {
  public:
    class TreeElem;
    TreeVector(ctranAvlTree::TreeVector::TreeElem *elem);
    ~TreeVector(void);
    void updateHeight(void);

    TreeVector *leftRotate(void);
    TreeVector *rightRotate(void);
    TreeVector *balance(void);
    TreeVector *insert(ctranAvlTree::TreeVector::TreeElem *elem);
    TreeVector *remove(void);

    uintptr_t addr;
    std::vector<ctranAvlTree::TreeVector::TreeElem *> avlTreeElemList;
    TreeVector *left;
    TreeVector *right;

  private:
    uint32_t height;
    void print(int level);
};

class ctranAvlTree::TreeVector::TreeElem {
  public:
    uintptr_t addr;
    std::size_t len;
    void *val;

    class ctranAvlTree::TreeVector *vec;
};

#endif
