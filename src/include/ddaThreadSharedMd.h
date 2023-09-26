// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef DDA_THREAD_SHARED_MD_H_
#define DDA_THREAD_SHARED_MD_H_

#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include "checks.h"

/*
 * We maintain three classes here:
 *
 * ddaPrivateMd -- this is local to each communicator and contains
 * rank-private information.
 *
 * ddaThreadSharedMd -- this is thread-shared metadata; all threaded
 * ranks within this communicator on this node share this information.
 *
 * ddaCliqueSharedMd -- this is meta-data that is shared within each
 * clique (mesh-connected GPUs) of threaded ranks.
 */

/* each clique (direct NVLink connected group) of ranks */
class ddaCliqueSharedMd {
 public:
  ddaCliqueSharedMd(std::vector<int> gpuClique) {
    this->gpus = gpuClique;

    /* mailbox for ranks to exchange their source buffer
     * information.  We create two copies and swap between the two
     * each time the collective is called.  This way, if one rank
     * is delayed in completing its work, we don't overwrite this
     * data in the next iteration. */
    for (int i = 0; i < 2; i++) {
      this->barrierMbox[i] = nullptr;
    }
  }

  ~ddaCliqueSharedMd() {
    for (int i = 0; i < 2; i++) {
      if (this->barrierMbox[i] != nullptr) {
        CUDACHECKIGNORE(cudaFree(this->barrierMbox[i]));
      }

      for (auto it : this->rankToLocalMbox[i]) {
        void* buf = it.second;
        CUDACHECKIGNORE(cudaFree(buf));
      }
      this->rankToLocalMbox[i].clear();
    }
  }

  void insertRank(int rank, int cudaDev) {
    auto it = std::find(this->gpus.begin(), this->gpus.end(), cudaDev);
    if (it == this->gpus.end()) {
      return;
    }

    if (this->rankToGpu.empty()) {
      int numBarrierPtrs = 2 * this->gpus.size();

      for (int i = 0; i < 2; i++) {
        CUDACHECKIGNORE(cudaMalloc(
            &this->barrierMbox[i], numBarrierPtrs * sizeof(uintptr_t)));
        CUDACHECKIGNORE(cudaMemset(
            this->barrierMbox[i], 0, numBarrierPtrs * sizeof(uintptr_t)));
      }
    }

    this->rankToGpu[rank] = cudaDev;

    for (int i = 0; i < 2; i++) {
      void* buf;

      CUDACHECKIGNORE(cudaMalloc(&buf, sizeof(uintptr_t)));
      CUDACHECKIGNORE(cudaMemset(buf, 0, sizeof(uintptr_t)));

      this->rankToLocalMbox[i][rank] = reinterpret_cast<uintptr_t*>(buf);
    }
  }

  bool searchRank(int rank) {
    auto got = this->rankToGpu.find(rank);
    return (got != this->rankToGpu.end());
  }

  /* mapping from rank to the GPU ID, temporary buffer, and local mbox */
  std::vector<int> gpus;
  std::unordered_map<int, int> rankToGpu;
  std::unordered_map<int, uintptr_t*> rankToLocalMbox[2];
  uintptr_t* barrierMbox[2];
};

/* metadata for dda ranks: contains the clique of GPUs (currently
 * all of the GPUs in the system), and a refcount of the number of
 * communicator handles in this address space that point to the same
 * commId */
class ddaThreadSharedMd {
 public:
  ddaThreadSharedMd(ncclUniqueId commId, std::vector<std::vector<int>> gpuCliques) {
    this->commId = commId;

    for (auto it : gpuCliques) {
      ddaCliqueSharedMd* clique = new ddaCliqueSharedMd(it);
      this->cliques.push_back(clique);
    }

    this->refCount = 0;
  }

  ~ddaThreadSharedMd() {
    for (auto c : cliques) {
      delete c;
    }
  }

  void insertRank(int rank, int cudaDev) {
    for (auto it : cliques) {
      it->insertRank(rank, cudaDev);
    }
  }

  bool searchRank(int rank) {
    bool ret = false;
    for (auto it : cliques) {
      ret = it->searchRank(rank);
      if (ret == true) {
        break;
      }
    }
    return ret;
  }

  ncclUniqueId commId;
  std::vector<ddaCliqueSharedMd*> cliques;
  int refCount;
};

#endif
