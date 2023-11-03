// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <deque>
#include "ctranAvlTree.h"
#include "ctranAvlTreeElem.h"
#include "debug.h"

ctranAvlTree::ctranAvlTree() {
  this->root = nullptr;
}

ctranAvlTree::~ctranAvlTree() {
  if (this->root) {
    delete this->root;
  }
}

void ctranAvlTree::insert(const void *addr_, std::size_t len, void *val, void **hdl) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(addr_);

  if (this->root == nullptr) {
    this->root = new ctranAvlTree::TreeElem(addr, len, val);
    *hdl = this->root;
  } else {
    this->root = this->root->insert(addr, len, val, (ctranAvlTree::TreeElem **) hdl);
    if (*hdl == nullptr) {
      auto e = new ctranAvlTree::TreeElem(addr, len, val);
      this->list.push_back(e);
      *hdl = e;
    }
  }
}

void ctranAvlTree::remove(void *hdl) {
  ctranAvlTree::TreeElem *e = reinterpret_cast<ctranAvlTree::TreeElem *>(hdl);

  auto it = std::find(this->list.begin(), this->list.end(), e);
  if (it != this->list.end()) {
    this->list.erase(std::remove(this->list.begin(), this->list.end(), e), this->list.end());
    goto exit;
  }

  if (this->root == e) {
    this->root = this->root->remove();
  } else {
    ctranAvlTree::TreeElem *r = this->root;
    std::vector<ctranAvlTree::TreeElem *> updateList;
    while (1) {
      if (r->left == e) {
        r->left = r->left->remove();
        updateList.push_back(r);
        break;
      } else if (r->right == e) {
        r->right = r->right->remove();
        updateList.push_back(r);
        break;
      } else if (r->addr > e->addr) {
        updateList.push_back(r);
        r = r->left;
      } else {
        updateList.push_back(r);
        r = r->right;
      }
    }
    for (auto e : updateList) {
      e->updateHeight();
    }
  }

exit:
  delete e;
}

void ctranAvlTree::search(const void *addr_, std::size_t len, void **hdl) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(const_cast<void *>(addr_));
  ctranAvlTree::TreeElem *r;

  for (auto e : this->list) {
    if (e->addr <= addr && e->addr + e->len >= addr + len) {
      *hdl = e;
      return;
    }
  }

  r = this->root;
  while (1) {
    if (r == nullptr) {
      *hdl = nullptr;
      break;
    } else if (r->addr > addr) {
      r = r->left;
      continue;
    } else if (r->addr + r->len < addr + len) {
      r = r->right;
      continue;
    } else {
      *hdl = r;
      break;
    }
  }
}

void ctranAvlTree::lookup(void *hdl, void **val) {
  *val = reinterpret_cast<ctranAvlTree::TreeElem *>(hdl)->val;
}

std::vector<void *> ctranAvlTree::getAllElems() {
  std::vector<void *> ret;
  std::deque<ctranAvlTree::TreeElem *> pendingList;

  for (auto e : this->list) {
    ret.push_back(e->val);
  }

  if (this->root != nullptr) {
    pendingList.push_back(this->root);
  }

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

  return ret;
}
