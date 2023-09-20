#ifndef NCCL_COLLTRACE_H_
#define NCCL_COLLTRACE_H_

#ifdef ENABLE_COLLTRACE

#include <condition_variable>
#include <iostream>
#include <list>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <cassert>
#include <atomic>

#include <cuda_runtime.h>

#include "info.h"
#include "utils.h"
#include "trainer.h"

// CUDA event pointer w/ deleter
struct CudaEventDeleter {
  void operator() (cudaEvent_t e) {
    CUDACHECKIGNORE(cudaEventDestroy(e));
  }
};
using CudaEventPtr = std::unique_ptr<std::pointer_traits<cudaEvent_t>::element_type, CudaEventDeleter>;

// Event data structure
struct EventInfo {
  ncclInfo info;
  int64_t iteration;
  CudaEventPtr start;
  CudaEventPtr stop;
  cudaStream_t stream;

  EventInfo() = default;
  EventInfo(const EventInfo&) = delete;
  EventInfo& operator=(const EventInfo&) = delete;
};

// Result data structure
struct ResultInfo {
  ncclInfo info;
  cudaStream_t stream;
  int64_t iteration;
  float latency;
};

// event pool
class SharedPool {
public:
  ~SharedPool(){
    while(!empty()){
      CudaEventPtr item = takeOne();
      CUDACHECKIGNORE(cudaEventDestroy(item.get()));
    }
  }

  void add(CudaEventPtr item) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push(std::move(item));
  }

  CudaEventPtr takeOne() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(pool_.empty()){
      cudaEvent_t newEvent = nullptr;
      CUDACHECKIGNORE(cudaEventCreate(&newEvent));
      CudaEventPtr item(newEvent);
      return item;
    }
    assert(!pool_.empty());
    CudaEventPtr tmp = std::move(pool_.front());
    pool_.pop();
    return tmp;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.size();
  }

private:
  std::queue<CudaEventPtr> pool_;
  mutable std::mutex mutex_;
};

// Class for colltrace
class CollTrace {
 private:
  // Work queue data structure
  class EventQueue {
   private:
    std::queue<std::unique_ptr<EventInfo>> queue_;
    std::mutex mutex_;

   public:
    void push(std::unique_ptr<EventInfo> item) {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(std::move(item));
    }
    bool isEmpty(){
      std::lock_guard<std::mutex> lock(mutex_);
      return queue_.empty();
    }
    std::unique_ptr<EventInfo> tryPop() {
      std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
      if (!lock.owns_lock()) {
        return NULL;
      }
      if (queue_.empty()) {
        return NULL;
      }
      std::unique_ptr<EventInfo> item = std::move(queue_.front());
      queue_.pop();

      return item;
    }
  };

  // Internal methods
  void outputResults();

  // Internal values
  SharedPool eventPool_;
  EventQueue eventQueue_;
  std::list<ResultInfo> results_;
  std::atomic<bool> workerThreadExitSignal_ { false };

  int rank_{-1};
  std::thread profilingWorkerThread_;

 public:

  CollTrace() = default;

  CollTrace(const CollTrace& obj) = delete;

  void* measureLatency();

  static void* measureLatencyWrapper(CollTrace* collTrace);

  ncclResult_t startWorkerThread(int rank);

  std::unique_ptr<EventInfo> getEventFromPool();

  ncclResult_t enqueueEvent(std::unique_ptr<EventInfo> eventInfo);

  ncclResult_t exit();
};
// Macros for comm.h if CollTrace is enabled
#define COLLTRACE_OBJECT() CollTrace* colltrace
// Macros for init.cc if CollTrace is enabled
#define COLLTRACE_INIT(comm) do{ \
                         comm->colltrace = new CollTrace(); \
                         NCCLCHECK(comm->colltrace->startWorkerThread(comm->rank)); \
                       } while(0)
#define COLLTRACE_EXIT(comm) comm->colltrace->exit()
// Macros for enqueue.cc if CollTrace is enabled
#define COLLTRACE_GET_TRAINING_ITERATION() getTrainingIteration()
#define COLLTRACE_INFO_COPY(plan, aggInfo) memcpy(&plan->aggInfo, &aggInfo, sizeof(ncclInfo))
#define COLLTRACE_ACQUIRE_EVENT(comm) std::unique_ptr<EventInfo> eventInfo; \
                                  do { \
                                    eventInfo = comm->colltrace->getEventFromPool(); \
                                    if(!eventInfo) { \
                                      return ncclInternalError; /*Event init failed*/ \
                                    } \
                                    eventInfo->iteration = getTrainingIteration(); \
                                  } while(0)
#define COLLTRACE_RECORD_START_EVENT() CUDACHECK(cudaEventRecord(eventInfo->start.get(), launchStream))
#define COLLTRACE_RECORD_END_EVENT(comm) do{ \
                                     CUDACHECK(cudaEventRecord(eventInfo->stop.get(), launchStream)); \
                                     eventInfo->info = plan->aggInfo; \
                                     eventInfo->stream = launchStream; \
                                     comm->colltrace->enqueueEvent(std::move(eventInfo)); \
                                   } while(0)
#else
// Define macros as empty is CollTrace is disabled
#define COLLTRACE_OBJECT()
#define COLLTRACE_INIT(comm)
#define COLLTRACE_EXIT(comm)
#define COLLTRACE_GET_TRAINING_ITERATION() 0
#define COLLTRACE_INFO_COPY(plan, aggInfo)
#define COLLTRACE_ACQUIRE_EVENT(comm)
#define COLLTRACE_RECORD_START_EVENT()
#define COLLTRACE_RECORD_END_EVENT(comm)
#endif // ENABLE_COLLTRACE
#endif // NCCL_COLLTRACE_H_
