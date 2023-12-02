// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CtranAvlTree.h"
#include <unistd.h>
#include <algorithm>
#include <cstddef>
#include <deque>
#include <iostream>
#include <sstream>
#include "CtranAvlTreeElem.h"
#include "debug.h"

CtranAvlTree::~CtranAvlTree() {
  if (this->root_) {
    delete this->root_;
  }
}

void* CtranAvlTree::insert(const void* addr_, std::size_t len, void* val) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(addr_);
  CtranAvlTree::TreeElem* hdl = nullptr;

  if (this->root_ == nullptr) {
    this->root_ = new CtranAvlTree::TreeElem(addr, len, val);
    hdl = this->root_;
  } else {
    // Try to insert into AVL tree for fast search
    this->root_ = this->root_->insert(addr, len, val, &hdl);

    // If fails to be handled by AVL tree, insert into list as fallback
    if (hdl == nullptr) {
      auto e = new CtranAvlTree::TreeElem(addr, len, val);
      this->list_.push_back(e);
      hdl = e;
    }
  }
  return hdl;
}

void CtranAvlTree::remove(void* hdl) {
  CtranAvlTree::TreeElem* e = reinterpret_cast<CtranAvlTree::TreeElem*>(hdl);

  // First try to remove from AVL tree
  bool removed = false;
  this->root_ = this->root_->remove(e, &removed);
  if (removed) {
    delete e;
  } else {
    // Not found in AVL tree, thus try to remove from list
    auto it = std::find(this->list_.begin(), this->list_.end(), e);
    if (it != this->list_.end()) {
      this->list_.erase(
          std::remove(this->list_.begin(), this->list_.end(), e),
          this->list_.end());
      delete e;
    }
  }
}

void* CtranAvlTree::search(const void* addr_, std::size_t len) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(const_cast<void*>(addr_));

  // First try to search in AVL tree
  CtranAvlTree::TreeElem* r = this->root_;
  while (r) {
    if (r->addr > addr) {
      r = r->left;
      continue;
    } else if (r->addr + r->len < addr + len) {
      r = r->right;
      continue;
    } else {
      // Found the matching node
      break;
    }
  }

  // If not found in AVL tree, search in list
  if (!r) {
    for (auto e : this->list_) {
      if (e->addr <= addr && e->addr + e->len >= addr + len) {
        r = e;
        break;
      }
    }
  }
  return r;
}

void* CtranAvlTree::lookup(void* hdl) {
  return reinterpret_cast<CtranAvlTree::TreeElem*>(hdl)->val;
}

std::vector<void*> CtranAvlTree::getAllElems() {
  std::vector<void*> ret;
  std::deque<CtranAvlTree::TreeElem*> pendingList;

  if (this->root_ != nullptr) {
    pendingList.push_back(this->root_);
  }

  // Enqueue all element in the tree via breadth first traversal
  while (!pendingList.empty()) {
    auto r = pendingList.front();
    pendingList.pop_front();
    ret.push_back(r->val);

    if (r->left) {
      pendingList.push_back(r->left);
    }
    if (r->right) {
      pendingList.push_back(r->right);
    }
  }

  for (auto e : this->list_) {
    ret.push_back(e->val);
  }

  return ret;
}

std::string CtranAvlTree::rangeToString(const void* addr, std::size_t len) {
  std::stringstream ss;
  ss << "[" << addr << ", " << len << "]";
  return ss.str();
}

std::string CtranAvlTree::toString() {
  std::stringstream ss;

  ss << "Internal AVL tree:" << std::endl;
  if (this->root_ != nullptr) {
    this->root_->treeToString(0, ss);
    ss << std::endl;
  }

  ss << "Internal list:" << std::endl;
  bool first = true;
  for (auto e : this->list_) {
    if (!first) {
      ss << ",";
    }
    ss << this->rangeToString(reinterpret_cast<const void*>(e->addr), e->len);
    first = false;
  }
  ss << std::endl;
  return ss.str();
}

size_t CtranAvlTree::size() {
  size_t size = 0;
  if (this->root_) {
    size += this->root_->size();
  }
  size += this->list_.size();
  return size;
}

bool CtranAvlTree::validateHeight() {
  // If root is null, it is balanced
  if (!this->root_) {
    return true;
  }

  // Check if all tree elements are with correct height
  return this->root_->validateHeight();
}

bool CtranAvlTree::isBalanced() {
  // If root is null, it is balanced
  if (!this->root_) {
    return true;
  }

  // Check if the tree is balanced
  return this->root_->isBalanced();
}
