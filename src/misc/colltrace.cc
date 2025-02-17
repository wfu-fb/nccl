#ifdef ENABLE_COLLTRACE
#include "colltrace.h"
#include "FbInternal.h"
#include "comm.h"
#include "bootstrap.h"

#include <unistd.h>
#include <chrono>
#include <sstream>
#include <string>
#include <fstream>

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_COLLTRACE_DIR
   type        : string
   default     : ""
   description : |-
     Directory for CollTrace to dump.
     Can be either local or FB internal remote URL.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

#ifndef COLLTRACE_IO_FB_DURING_RUN
#define COLLTRACE_IO_FB_DURING_RUN
#endif

void CollTrace::outputResults(){
  std::stringstream stream;
  stream << "[\n  {\n";
  for (auto it = results_.begin(); it != results_.end(); ++it) {
    if (it != results_.begin()) {
      stream << "  },\n  {\n";
    }
    stream << "    \"coll\": \"" << it->info.opName << "\",\n"
          << "    \"msg_size\": \""
          << (it->info.count * ncclTypeSize(it->info.datatype)) << "\",\n"
          << "    \"latency\": " << it->latency << "\n";
  }
  stream << "  }\n]";


  // If NCCL_COLLTRACE_DIR is set, then write profiling data to file
  if(!NCCL_COLLTRACE_DIR.empty()){
    const std::string fileName =
        NCCL_COLLTRACE_DIR + "/" + std::to_string(rank_) + "_online.json";
    INFO(NCCL_ALL, "Rank %d: Writing %lu online profiler data to : %s", rank_, results_.size(), fileName.c_str());

    if (ncclIsFbPath(fileName)) {
      ncclFbUpload(stream.str(), fileName);
    } else {
      std::ofstream f(fileName);
      f << stream.str();
      f.close();
    }
  }
}

void* CollTrace::measureLatency() {
  INFO(NCCL_INIT, "Rank %d: Started CollTrace worker thread", rank_);

  while (true) {
    std::unique_ptr<EventInfo> curEvent = eventQueue_.tryPop();
    if(curEvent){
      if (curEvent->info.count != 0) {
        cudaError_t res = cudaEventSynchronize(curEvent->stop.get());
        float latency = -1;
        res = res == cudaSuccess ? cudaEventElapsedTime(&latency, curEvent->start.get(), curEvent->stop.get()) : res;
        ResultInfo result;
        result.info = curEvent->info;
        result.stream = curEvent->stream;
        if(res == cudaSuccess){
          result.latency = latency;
        }
        result.iteration = curEvent->iteration;
        results_.push_back(result);
        eventPool_.add(std::move(curEvent->start));
        eventPool_.add(std::move(curEvent->stop));
        COLLTRACE_IO_FB_DURING_RUN(result, rank_);

        if (curEvent->info.comm->tuner != NULL) {
          results_.push_back(result);

          // Online tuning - average latencies across ranks & send to tuner
          float* latencies = NULL;
          NCCLCHECKIGNORE(ncclCalloc(&latencies, curEvent->info.comm->nRanks));
          latencies[curEvent->info.comm->rank] = latency;
          NCCLCHECKIGNORE(bootstrapAllGather(curEvent->info.comm->bootstrap, latencies, sizeof(float)));
          float sum = 0.0;
          for(int i = 0; i < curEvent->info.comm->nRanks; i++){
            sum += latencies[i];
          }

          free(latencies);
          sum /= (float) curEvent->info.comm->nRanks;

          curEvent->info.comm->tuner->addOnlineResult(
            curEvent->info.coll,
            curEvent->info.count * ncclTypeSize(curEvent->info.datatype),
            curEvent->iteration,
            sum,
            curEvent->info.algorithm,
            curEvent->info.protocol,
            curEvent->info.nChannels,
            curEvent->info.nThreads);
        }

      }
    } else {
      if (workerThreadExitSignal_ && eventQueue_.isEmpty()) {
        outputResults();
        break;
      }
    }
  }

  return NULL;
}

void* CollTrace::measureLatencyWrapper(CollTrace* collTrace){
  return collTrace->measureLatency();
}

ncclResult_t CollTrace::startWorkerThread(int rank) {
  // create worker thread
  rank_ = rank;
  profilingWorkerThread_ = std::thread{ measureLatencyWrapper, this };

  return ncclSuccess;
}

std::unique_ptr<EventInfo> CollTrace::getEventFromPool(){
  std::unique_ptr<EventInfo> eventInfo(new EventInfo);
  eventInfo->start = eventPool_.takeOne();
  eventInfo->stop = eventPool_.takeOne();
  if(!eventInfo->start || !eventInfo->stop){
    std::unique_ptr<EventInfo> nullEventInfo(nullptr);
    return nullEventInfo;
  }
  return eventInfo;
}

ncclResult_t CollTrace::enqueueEvent(std::unique_ptr<EventInfo> eventInfo) {
  eventQueue_.push(std::move(eventInfo));

  return ncclSuccess;
}

ncclResult_t CollTrace::exit() {
  workerThreadExitSignal_ = true;
  profilingWorkerThread_.join();

  return ncclSuccess;
}
#endif
