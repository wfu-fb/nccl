// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_AVL_TREE_H_
#define CTRAN_AVL_TREE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "nccl.h"

/**
 * AVL tree.
 * It supports both non-overlapping address ranges and overlapping address
 * ranges. Since most of the usecase would be non-overlapping ranges, we
 * optimize it using an intenral AVL tree structure (TreeElem* root_) which
 * provides O(logN) insert, search, and remove complexity. For any overlapping
 * ranges, we maintain them using a list (std::vector<TreeElem*> list_).
 */
class CtranAvlTree {
 public:
  CtranAvlTree() = default;
  ~CtranAvlTree();

  // Insert a new element into the tree and return the corresponding handle.
  // If the new element range overlaps with any existing element, insertion
  // fails and nullptr is returned.
  void* insert(const void* addr, std::size_t len, void* val);

  // Remove an element from the tree by searching the provided handle.
  ncclResult_t remove(void* hdl);

  // Search for an element in the tree, handle is returned if found; otherwise
  // return nullptr.
  void* search(const void* addr, std::size_t len);

  // Lookup the value of the provided handle.
  void* lookup(void* hdl);

  // format a given range to a string with consistent format
  static std::string rangeToString(const void* addr, std::size_t len);

  // Print all elements in the tree into a string.
  std::string toString();

  // Get all elements in the tree.
  std::vector<void*> getAllElems();

  // Get total number of elements.
  size_t size();

  // Validate if all elements in the tree is with the correct height.
  bool validateHeight();

  // Check all nodes in the tree are balanced (i.e., the height difference of
  // left and right sub trees is <= 1)
  bool isBalanced();

 private:
  class TreeElem;
  class TreeElem* root_;
  std::vector<TreeElem*> list_;
};

#endif
