// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "nccl.h"
#include "ctranAvlTree.h"
#include "../../../include/checks.h"
#include <gtest/gtest.h>
#include <iostream>
#include <unordered_set>

#define MAX_BUFS (1024 * 1024)
#define MAX_LEN  (1024)
#define ITERS    (10000)

struct bufMd {
  void *addr;
  std::size_t len;
  void *val;
  void *hdl;
};

TEST(AvlTreeTest, Test) {
  ncclResult_t res = ncclSuccess;
  ctranAvlTree *tree = new ctranAvlTree();

  srand(0);
  std::vector<struct bufMd> bufList;
  std::unordered_set<uintptr_t> addrSet;
  for (int i = 0; i < MAX_BUFS; i++) {
    struct bufMd buf;
    buf.addr = reinterpret_cast<void *>(((rand() % UINTPTR_MAX) / MAX_LEN) * MAX_LEN);
    buf.len = rand() % MAX_LEN;
    buf.val = reinterpret_cast<void *>(static_cast<uintptr_t>(i));

    if (addrSet.find(reinterpret_cast<uintptr_t>(buf.addr)) != addrSet.end()) {
      continue;
    } else {
      addrSet.insert(reinterpret_cast<uintptr_t>(buf.addr));
    }

    NCCLCHECKGOTO(tree->insert(buf.addr, buf.len, buf.val, &buf.hdl), res, exit);
    bufList.push_back(buf);
  }
  addrSet.clear();

  for (int i = 0; i < ITERS; i++) {
    void *hdl, *val;
    int idx = rand() % bufList.size();
    tree->search(bufList[idx].addr, bufList[idx].len, &hdl);
    tree->lookup(hdl, &val);

    EXPECT_EQ(bufList[idx].hdl, hdl);
    EXPECT_EQ(bufList[idx].val, val);
  }

  delete tree;

exit:
  return;
}
