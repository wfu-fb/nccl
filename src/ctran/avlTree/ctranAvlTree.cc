// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <deque>
#include "ctranAvlTree.h"
#include "ctranAvlTreeVector.h"
#include "debug.h"

ctranAvlTree::ctranAvlTree() {
  this->root = nullptr;
}

ctranAvlTree::~ctranAvlTree() {
  if (this->root) {
    delete this->root;
  }
}

ncclResult_t ctranAvlTree::insert(const void *addr, std::size_t len, void *val, void **hdl) {
  ctranAvlTree::TreeVector::TreeElem *elem = new ctranAvlTree::TreeVector::TreeElem;
  elem->addr = reinterpret_cast<uintptr_t>(addr);
  elem->len = len;
  elem->val = val;
  *hdl = reinterpret_cast<void *>(elem);

  if (this->root == nullptr) {
    this->root = new ctranAvlTree::TreeVector(elem);
  } else {
    this->root = this->root->insert(elem);
  }

  return ncclSuccess;
}

ncclResult_t ctranAvlTree::remove(void *hdl) {
  ncclResult_t res = ncclSuccess;
  ctranAvlTree::TreeVector::TreeElem *elem = reinterpret_cast<ctranAvlTree::TreeVector::TreeElem *>(hdl);
  ctranAvlTree::TreeVector *vec = elem->vec;

  bool found = false;
  for (auto it = vec->avlTreeElemList.begin(); it != vec->avlTreeElemList.end(); it++) {
    if (*it == elem) {
      vec->avlTreeElemList.erase(it);
      delete elem;
      found = true;
      break;
    }
  }
  if (found == false) {
    goto fail;
  }

  if (!vec->avlTreeElemList.empty()) {
    goto exit;
  }

  if (this->root == vec) {
    this->root = vec->remove();
  } else {
    ctranAvlTree::TreeVector *r = this->root;
    while (1) {
      if (r->left && r->left->addr == vec->addr) {
        r->left = vec->remove();
        r->updateHeight();
        break;
      } else if (r->right && r->right->addr == vec->addr) {
        r->right = vec->remove();
        r->updateHeight();
        break;
      } else if (r->addr > vec->addr) {
        r = r->left;
      } else {
        r = r->right;
      }
    }
  }
  delete vec;

exit:
  return res;

fail:
  res = ncclSystemError;
  goto exit;
}

ncclResult_t ctranAvlTree::search(const void *addr_, std::size_t len, void **hdl) {
  ncclResult_t res = ncclSuccess;
  uintptr_t addr = reinterpret_cast<uintptr_t>(const_cast<void *>(addr_));

  ctranAvlTree::TreeVector *r = this->root;
  while (1) {
    if (r == nullptr) {
      *hdl = nullptr;
      break;
    } else if (r->addr > addr) {
      r = r->left;
      continue;
    } else if (r->right && r->right->addr <= addr) {
      r = r->right;
      continue;
    } else {
      for (auto e : r->avlTreeElemList) {
        if (e->addr <= addr && e->addr + e->len >= addr + len) {
          *hdl = e;
          goto exit;
        }
      }
    }
  }

exit:
  return res;
}

ncclResult_t ctranAvlTree::lookup(void *hdl, void **val) {
  *val = reinterpret_cast<ctranAvlTree::TreeVector::TreeElem *>(hdl)->val;

  return ncclSuccess;
}

std::vector<void *> ctranAvlTree::getAllElems() {
  std::vector<void *> ret;
  std::vector<ctranAvlTree::TreeVector *> traversedList;
  std::deque<ctranAvlTree::TreeVector *> pendingList;

  if (this->root == nullptr) {
    goto exit;
  }

  pendingList.push_back(this->root);
  while (!pendingList.empty()) {
    auto r = pendingList.front();
    pendingList.pop_front();
    traversedList.push_back(r);
    if (r->left) {
      pendingList.push_back(r->left);
    }
    if (r->right) {
      pendingList.push_back(r->right);
    }
  }

  for (auto v : traversedList) {
    for (auto e : v->avlTreeElemList) {
      ret.push_back(e->val);
    }
  }

exit:
  return ret;
}
