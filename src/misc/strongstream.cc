/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "strongstream.h"
#include "cudawrap.h"
#include "checks.h"
#include "param.h"
#include "nccl_cvars.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_GRAPH_MIXING_SUPPORT
   type        : bool
   default     : true
   description : |-
     Enable/disable support for co-occurring outstanding NCCL launches
     from multiple CUDA graphs or a CUDA graph and non-captured NCCL
     calls. With support disabled, correctness is only guaranteed if
     the communicator always avoids both of the following cases:
     1. Has outstanding parallel graph launches, where parallel
     means on different streams without dependencies that would
     otherwise serialize their execution.
     2. An outstanding graph launch followed by a non-captured
     launch.  Stream dependencies are irrelevant.
     For more information:
     https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-mixing-support

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

// Tracks the chain of graph nodes for a given graph captured identified by
// its graph id. This state has to live for as long as captured work is being
// submitted. CUDA doesn't have mechanism to inform us when the user ends capture
// so the best we can do is get notified when the graph is destroyed.
struct ncclStrongStreamGraph {
  struct ncclStrongStreamGraph* next;
  // Atomically exchanged to false by both the main thread or the graph destructor
  // callback. The last to arrive deletes the node.
  bool alive;
  unsigned long long graphId;
  // For each graph we track the "tip" of the chain of graph nodes. A linear
  // chain would always have just one node at its tip, but since we have to merge
  // in chains from other streams (via ncclStrongStreamWaitStream) some spots
  // in the chain can be wider than a single node and thus need a list, so we
  // maintain a dynamically sized array of tip nodes.
  int tipCount, tipCapacity;
  cudaGraphNode_t* tipNodes;
};

static void ncclStrongStreamGraphDelete(struct ncclStrongStreamGraph* g) {
  free(g->tipNodes);
  free(g);
}

////////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclCudaGetCapturingGraph(
    struct ncclCudaGraph* graph, cudaStream_t stream
  ) {
  #if CUDART_VERSION >= 10000 // cudaStreamGetCaptureInfo
    int driver;
    NCCLCHECK(ncclCudaDriverVersion(&driver));
    if (CUDART_VERSION < 11030 || driver < 11030) {
      cudaStreamCaptureStatus status;
      unsigned long long gid;
      CUDACHECK(cudaWrapper->cudaStreamGetCaptureInfo(stream, &status, &gid));
      #if CUDART_VERSION >= 11030
        graph->graph = nullptr;
        graph->graphId = ULLONG_MAX;
      #endif
      if (status != cudaStreamCaptureStatusNone) {
        WARN("NCCL cannot be captured in a graph if either it wasn't built with CUDA runtime >= 11.3 or if the installed CUDA driver < R465.");
        return ncclInvalidUsage;
      }
    } else {
      #if CUDART_VERSION >= 11030
        cudaStreamCaptureStatus status;
        unsigned long long gid;
        CUDACHECK(cudaWrapper->cudaStreamGetCaptureInfo_v2(stream, &status, &gid, &graph->graph, nullptr, nullptr));
        if (status != cudaStreamCaptureStatusActive) {
          graph->graph = nullptr;
          gid = ULLONG_MAX;
        }
        graph->graphId = gid;
      #endif
    }
  #endif
  return ncclSuccess;
}

ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph graph, cudaHostFn_t fn, void* arg) {
  #if CUDART_VERSION >= 11030
    cudaUserObject_t object;
    CUDACHECK(cudaWrapper->cudaUserObjectCreate(
      &object, arg, fn, /*initialRefcount=*/1, cudaUserObjectNoDestructorSync
    ));
    // Hand over ownership to CUDA Graph
    CUDACHECK(cudaWrapper->cudaGraphRetainUserObject(graph.graph, object, 1, cudaGraphUserObjectMove));
    return ncclSuccess;
  #else
    return ncclInvalidUsage;
  #endif
}

////////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream* ss) {
  CUDACHECK(cudaWrapper->cudaStreamCreateWithFlags(&ss->cudaStream, cudaStreamNonBlocking));
  #if CUDART_VERSION >= 11030
    CUDACHECK(cudaWrapper->cudaEventCreateWithFlags(&ss->serialEvent, cudaEventDisableTiming));
    ss->everCaptured = false;
    ss->serialEventNeedsRecord = false;
    ss->graphHead = nullptr;
  #else
    CUDACHECK(cudaWrapper->cudaEventCreateWithFlags(&ss->scratchEvent, cudaEventDisableTiming));
  #endif
  return ncclSuccess;
}

static void graphDestructor(void* arg) {
  struct ncclStrongStreamGraph* g = (struct ncclStrongStreamGraph*)arg;
  if (false == __atomic_exchange_n(&g->alive, false, __ATOMIC_ACQ_REL)) {
    // Last to arrive deletes list node.
    ncclStrongStreamGraphDelete(g);
  }
}

ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss) {
  CUDACHECK(cudaWrapper->cudaStreamDestroy(ss->cudaStream));
  #if CUDART_VERSION >= 11030
    CUDACHECK(cudaWrapper->cudaEventDestroy(ss->serialEvent));
    // Delete list of per-graph chains.
    struct ncclStrongStreamGraph* g = ss->graphHead;
    while (g != nullptr) {
      struct ncclStrongStreamGraph* next = g->next;
      if (false == __atomic_exchange_n(&g->alive, false, __ATOMIC_ACQ_REL)) {
        // Last to arrive deletes list node.
        ncclStrongStreamGraphDelete(g);
      }
      g = next;
    }
  #else
    CUDACHECK(cudaWrapper->cudaEventDestroy(ss->scratchEvent));
  #endif
  return ncclSuccess;
}

static void ensureTips(struct ncclStrongStreamGraph* g, int n) {
  if (g->tipCapacity < n) {
    g->tipNodes = (cudaGraphNode_t*)realloc(g->tipNodes, n*sizeof(cudaGraphNode_t));
    g->tipCapacity = n;
  }
}

ncclResult_t ncclStrongStreamAcquire(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      if (NCCL_GRAPH_MIXING_SUPPORT && ss->everCaptured) {
        CUDACHECK(cudaWrapper->cudaStreamWaitEvent(ss->cudaStream, ss->serialEvent, 0));
        ss->serialEventNeedsRecord = false;
      }
    } else {
      ss->everCaptured = true;
      // Find the current graph in our list of graphs if it exists.
      struct ncclStrongStreamGraph** pg = &ss->graphHead;
      struct ncclStrongStreamGraph* g;
      while (*pg != nullptr) {
        g = *pg;
        if (g->graphId == graph.graphId) {
          // Move to front of list so that operations after acquire don't have to search the list.
          *pg = g->next;
          g->next = ss->graphHead;
          ss->graphHead = g;
          return ncclSuccess;
        } else if (false == __atomic_load_n(&g->alive, __ATOMIC_ACQUIRE)) {
          // Unrelated graph that has been destroyed. Remove and delete.
          *pg = g->next;
          ncclStrongStreamGraphDelete(g);
        } else {
          pg = &g->next;
        }
      }

      // This is a new graph so add to the list.
      g = (struct ncclStrongStreamGraph*)malloc(sizeof(struct ncclStrongStreamGraph));
      g->graphId = graph.graphId;
      g->tipNodes = nullptr;
      g->tipCapacity = 0;
      g->tipCount = 0;
      g->next = ss->graphHead;
      ss->graphHead = g;
      g->alive = true;
      NCCLCHECK(ncclCudaGraphAddDestructor(graph, graphDestructor, (void*)g));

      if (NCCL_GRAPH_MIXING_SUPPORT && ss->serialEventNeedsRecord) {
        // Can only be here if previous release was for uncaptured work that
        // elided updating the event because no capture had yet occurred.
        CUDACHECK(cudaWrapper->cudaStreamWaitEvent(ss->cudaStream, ss->serialEvent, 0));
        CUDACHECK(cudaWrapper->cudaEventRecord(ss->serialEvent, ss->cudaStream));
      }
      ss->serialEventNeedsRecord = false;

      // First node in the chain must be a wait on the serialEvent.
      if (NCCL_GRAPH_MIXING_SUPPORT) {
        ensureTips(g, 1);
        CUDACHECK(cudaWrapper->cudaGraphAddEventWaitNode(&g->tipNodes[0], graph.graph, nullptr, 0, ss->serialEvent));
        g->tipCount = 1;
      } else {
        g->tipCount = 0;
      }
    }
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamAcquireUncaptured(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    if (NCCL_GRAPH_MIXING_SUPPORT && ss->everCaptured) {
      CUDACHECK(cudaWrapper->cudaStreamWaitEvent(ss->cudaStream, ss->serialEvent, 0));
    }
    ss->serialEventNeedsRecord = true; // Assume the caller is going to add work to stream.
  #endif
  return ncclSuccess;
}

static ncclResult_t checkGraphId(struct ncclStrongStreamGraph* g, unsigned long long id) {
  if (g == nullptr || g->graphId != id) {
    WARN("Expected graph id=%llu was not at head of strong stream's internal list.", id);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    if (NCCL_GRAPH_MIXING_SUPPORT && ss->serialEventNeedsRecord) {
      if (graph.graph == nullptr) {
        if (ss->everCaptured) {
          CUDACHECK(cudaWrapper->cudaEventRecord(ss->serialEvent, ss->cudaStream));
          ss->serialEventNeedsRecord = false;
        }
      } else {
        struct ncclStrongStreamGraph* g = ss->graphHead;
        NCCLCHECK(checkGraphId(g, graph.graphId));
        ensureTips(g, 1);
        CUDACHECK(cudaWrapper->cudaGraphAddEventRecordNode(&g->tipNodes[0], graph.graph, g->tipNodes, g->tipCount, ss->serialEvent));
        g->tipCount = 1;
        ss->serialEventNeedsRecord = false;
      }
    }
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamLaunchHost(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss, cudaHostFn_t fn, void* arg
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      CUDACHECK(cudaWrapper->cudaLaunchHostFunc(ss->cudaStream, fn, arg));
    } else {
      cudaHostNodeParams p;
      p.fn = fn;
      p.userData = arg;
      struct ncclStrongStreamGraph* g = ss->graphHead;
      NCCLCHECK(checkGraphId(g, graph.graphId));
      ensureTips(g, 1);
      CUDACHECK(cudaWrapper->cudaGraphAddHostNode(&g->tipNodes[0], graph.graph, g->tipNodes, g->tipCount, &p));
      g->tipCount = 1;
    }
    ss->serialEventNeedsRecord = true;
  #else
    CUDACHECK(cudaWrapper->cudaLaunchHostFunc(ss->cudaStream, fn, arg));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamLaunchKernel(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss,
    void* fn, dim3 grid, dim3 block, void* args[], size_t sharedMemBytes
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      CUDACHECK(cudaWrapper->cudaLaunchKernel(fn, grid, block, args, sharedMemBytes, ss->cudaStream));
    } else {
      cudaKernelNodeParams p;
      p.func = fn;
      p.gridDim = grid;
      p.blockDim = block;
      p.kernelParams = args;
      p.sharedMemBytes = sharedMemBytes;
      p.extra = nullptr;
      struct ncclStrongStreamGraph* g = ss->graphHead;
      NCCLCHECK(checkGraphId(g, graph.graphId));
      ensureTips(g, 1);
      CUDACHECK(cudaWrapper->cudaGraphAddKernelNode(&g->tipNodes[0], graph.graph, g->tipNodes, g->tipCount, &p));
      g->tipCount = 1;
    }
    ss->serialEventNeedsRecord = true;
  #else
    CUDACHECK(cudaWrapper->cudaLaunchKernel(fn, grid, block, args, sharedMemBytes, ss->cudaStream));
  #endif
  return ncclSuccess;
}

// Merge node list `b` into list `a` but don't add duplicates.
static void mergeTips(struct ncclStrongStreamGraph* a, cudaGraphNode_t const* bNodes, int bn) {
  int an = a->tipCount;
  ensureTips(a, an + bn);
  for (int bi=0; bi < bn; bi++) {
    for (int ai=0; ai < an; ai++) {
      if (a->tipNodes[ai] == bNodes[bi]) goto next_b;
    }
    a->tipNodes[a->tipCount++] = bNodes[bi];
  next_b:;
  }
}

ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, struct ncclStrongStream* a, struct ncclStrongStream* b,
    bool b_subsumes_a
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      if (b->serialEventNeedsRecord) {
        b->serialEventNeedsRecord = false;
        CUDACHECK(cudaWrapper->cudaEventRecord(b->serialEvent, b->cudaStream));
      }
      CUDACHECK(cudaWrapper->cudaStreamWaitEvent(a->cudaStream, b->serialEvent, 0));
    } else {
      struct ncclStrongStreamGraph* ag = a->graphHead;
      NCCLCHECK(checkGraphId(ag, graph.graphId));
      struct ncclStrongStreamGraph* bg = b->graphHead;
      NCCLCHECK(checkGraphId(bg, graph.graphId));
      if (b_subsumes_a) ag->tipCount = 0;
      mergeTips(ag, bg->tipNodes, bg->tipCount);
    }
    a->serialEventNeedsRecord = true;
  #else
    CUDACHECK(cudaWrapper->cudaEventRecord(b->scratchEvent, b->cudaStream));
    CUDACHECK(cudaWrapper->cudaStreamWaitEvent(a->cudaStream, b->scratchEvent, 0));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, struct ncclStrongStream* a, cudaStream_t b,
    bool b_subsumes_a
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      // It is ok to use a->serialEvent to record b since we'll be setting
      // a->serialEventNeedsRecord so the event won't be considered accurate
      // until re-recorded.
      CUDACHECK(cudaWrapper->cudaEventRecord(a->serialEvent, b));
      CUDACHECK(cudaWrapper->cudaStreamWaitEvent(a->cudaStream, a->serialEvent, 0));
    } else {
      cudaStreamCaptureStatus status;
      unsigned long long bGraphId;
      cudaGraphNode_t const* bNodes;
      size_t bCount = 0;
      CUDACHECK(cudaWrapper->cudaStreamGetCaptureInfo_v2(b, &status, &bGraphId, nullptr, &bNodes, &bCount));
      if (status != cudaStreamCaptureStatusActive || graph.graphId != bGraphId) {
        WARN("Stream is not being captured by the expected graph.");
        return ncclInvalidUsage;
      }
      struct ncclStrongStreamGraph* ag = a->graphHead;
      NCCLCHECK(checkGraphId(ag, graph.graphId));
      if (b_subsumes_a) ag->tipCount = 0;
      mergeTips(ag, bNodes, bCount);
    }
    a->serialEventNeedsRecord = true;
  #else
    CUDACHECK(cudaWrapper->cudaEventRecord(a->scratchEvent, b));
    CUDACHECK(cudaWrapper->cudaStreamWaitEvent(a->cudaStream, a->scratchEvent, 0));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, cudaStream_t a, struct ncclStrongStream* b,
    bool b_subsumes_a
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      if (b->serialEventNeedsRecord) {
        b->serialEventNeedsRecord = false;
        CUDACHECK(cudaWrapper->cudaEventRecord(b->serialEvent, b->cudaStream));
      }
      CUDACHECK(cudaWrapper->cudaStreamWaitEvent(a, b->serialEvent, 0));
    } else {
      struct ncclStrongStreamGraph* bg = b->graphHead;
      NCCLCHECK(checkGraphId(bg, graph.graphId));
      CUDACHECK(cudaWrapper->cudaStreamUpdateCaptureDependencies(a, bg->tipNodes, bg->tipCount,
        b_subsumes_a ? cudaStreamSetCaptureDependencies : cudaStreamAddCaptureDependencies
      ));
    }
  #else
    CUDACHECK(cudaWrapper->cudaEventRecord(b->scratchEvent, b->cudaStream));
    CUDACHECK(cudaWrapper->cudaStreamWaitEvent(a, b->scratchEvent, 0));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    CUDACHECK(cudaWrapper->cudaStreamWaitEvent(ss->cudaStream, ss->serialEvent, 0));
    ss->serialEventNeedsRecord = false;
  #endif
  CUDACHECK(cudaWrapper->cudaStreamSynchronize(ss->cudaStream));
  return ncclSuccess;
}
