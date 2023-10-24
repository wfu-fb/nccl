// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <deque>
#include "ctranAvlTree.h"
#include "ctranAvlTreeVector.h"
#include "debug.h"

ctranAvlTree::TreeVector::TreeVector(ctranAvlTree::TreeVector::TreeElem *elem) {
  this->addr = elem->addr;
  this->height = 1;
  this->left = nullptr;
  this->right = nullptr;
  this->avlTreeElemList.push_back(elem);
  elem->vec = this;
}

ctranAvlTree::TreeVector::~TreeVector(void) {
  for (auto e : this->avlTreeElemList) {
    delete e;
  }
  if (this->left) {
    delete this->left;
  }
  if (this->right) {
    delete this->right;
  }
}

void ctranAvlTree::TreeVector::updateHeight(void) {
  uint32_t lHeight, rHeight;

  if (this->left == nullptr) {
    lHeight = 0;
  } else {
    lHeight = this->left->height;
  }

  if (this->right == nullptr) {
    rHeight = 0;
  } else {
    rHeight = this->right->height;
  }

  this->height = std::max(lHeight, rHeight) + 1;
}

void ctranAvlTree::TreeVector::print(int level) {
  if (level && this->left == nullptr && this->right == nullptr) {
    return;
  }

  for (int i = 0; i < level; i++) {
    printf("    ");
  }

  printf("%zu (%u) %zu (%u) %zu (%u)\n", this->addr, this->height,
      this->left ? this->left->addr : 0, this->left ? this->left->height : 0,
      this->right ? this->right->addr : 0, this->right ? this->right->height : 0);
  fflush(stdout);
  if (this->left) {
    this->left->print(level + 1);
  }
  if (this->right) {
    this->right->print(level + 1);
  }
}

ctranAvlTree::TreeVector *ctranAvlTree::TreeVector::leftRotate(void) {
  if (this->right == nullptr) {
    return nullptr;
  }

  ctranAvlTree::TreeVector *newroot = this->right;
  this->right = newroot->left;
  this->updateHeight();

  newroot->left = this;
  newroot->updateHeight();

  return newroot;
}

ctranAvlTree::TreeVector *ctranAvlTree::TreeVector::rightRotate(void) {
  if (this->left == nullptr) {
    return nullptr;
  }

  ctranAvlTree::TreeVector *newroot = this->left;
  this->left = newroot->right;
  this->updateHeight();

  newroot->right = this;
  newroot->updateHeight();

  return newroot;
}

ctranAvlTree::TreeVector *ctranAvlTree::TreeVector::balance(void) {
  uint32_t leftHeight = this->left ? this->left->height : 0;
  uint32_t rightHeight = this->right ? this->right->height : 0;

  if (leftHeight > rightHeight + 1) {
    uint32_t leftLeftHeight = this->left->left ? this->left->left->height : 0;
    uint32_t leftRightHeight = this->left->right ? this->left->right->height : 0;

    if (leftLeftHeight > leftRightHeight) {
      return this->rightRotate();
    } else {
      this->left = this->left->leftRotate();
      return this->rightRotate();
    }
  } else if (rightHeight > leftHeight + 1) {
    uint32_t rightLeftHeight = this->right->left ? this->right->left->height : 0;
    uint32_t rightRightHeight = this->right->right ? this->right->right->height : 0;

    if (rightRightHeight > rightLeftHeight) {
      return this->leftRotate();
    } else {
      this->right = this->right->rightRotate();
      return this->leftRotate();
    }
  }

  return this;
}

ctranAvlTree::TreeVector *ctranAvlTree::TreeVector::insert(ctranAvlTree::TreeVector::TreeElem *elem) {
  ctranAvlTree::TreeVector *newroot;

  if (this->addr > elem->addr) {
    if (this->left == nullptr) {
      this->left = new ctranAvlTree::TreeVector(elem);
    } else {
      this->left = this->left->insert(elem);
    }
    this->left->updateHeight();
    this->updateHeight();
    newroot = this->balance();
  } else if (this->addr < elem->addr) {
    if (this->right == nullptr) {
      this->right = new ctranAvlTree::TreeVector(elem);
    } else {
      this->right = this->right->insert(elem);
    }
    this->right->updateHeight();
    this->updateHeight();
    newroot = this->balance();
  } else {
    this->avlTreeElemList.push_back(elem);
    elem->vec = this;
    newroot = this;
  }

  return newroot;
}

ctranAvlTree::TreeVector *ctranAvlTree::TreeVector::remove(void) {
  ctranAvlTree::TreeVector *newroot;

  if (this->left == nullptr && this->right == nullptr) {
    newroot = nullptr;
  } else if (this->left == nullptr) {
    newroot = this->right;
  } else if (this->right == nullptr) {
    newroot = this->left;
  } else if (this->left->right == nullptr) {
    this->left->right = this->right;
    newroot = this->left;
  } else if (this->right->left == nullptr) {
    this->right->left = this->left;
    newroot = this->right;
  } else if (this->left->height >= this->right->height) {
    ctranAvlTree::TreeVector *tmp = this->left;
    while (tmp->right->right) {
      tmp = tmp->right;
    }
    newroot = tmp->right;
    tmp->right = tmp->right->remove();
    newroot->left = this->left;
    newroot->right = this->right;
  } else {
    ctranAvlTree::TreeVector *tmp = this->right;
    while (tmp->left->left) {
      tmp = tmp->left;
    }
    newroot = tmp->left;
    tmp->left = tmp->left->remove();
    newroot->left = this->left;
    newroot->right = this->right;
  }

  this->left = nullptr;
  this->right = nullptr;

  if (newroot) {
    newroot->updateHeight();
  }
  return newroot;
}
