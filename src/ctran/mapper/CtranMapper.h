// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_MAPPER_H_
#define CTRAN_MAPPER_H_

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include "CtranIb.h"
#include "checks.h"
#include "nccl.h"

struct CtranMapperRemoteAccessKey {
  struct CtranIbRemoteAccessKey ibKey;
};

class CtranMapper;
class CtranMapperRequest {
 public:
  CtranMapperRequest(CtranMapper* mapper);
  ~CtranMapperRequest();

  /* test whether a request is completed or not */
  ncclResult_t test(bool* isComplete);
  /* wait the complete of this request */
  ncclResult_t wait();

  CtranIbRequest* ibReq;

 private:
  CtranMapper* mapper_;
  enum {
    INCOMPLETE,
    COMPLETE,
  } state_;
};

struct ncclComm;

class CtranMapperTimestampPoint {
  public:
    CtranMapperTimestampPoint(int peer) {
      this->now = std::chrono::high_resolution_clock::now();
      this->peer = peer;
    }
    ~CtranMapperTimestampPoint() = default;

    std::chrono::time_point<std::chrono::high_resolution_clock> now;
    int peer;
};

class CtranMapperTimestamp {
  public:
    CtranMapperTimestamp(const std::string algo) {
      this->algo = algo;
      this->start = std::chrono::high_resolution_clock::now();
    }
    ~CtranMapperTimestamp() = default;

    std::vector<CtranMapperTimestampPoint> recvCtrl;
    std::vector<CtranMapperTimestampPoint> putIssued;
    std::vector<CtranMapperTimestampPoint> putComplete;
    std::string algo;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

class CtranMapperTimer {
 public:
  CtranMapperTimer() {
    this->start_ = std::chrono::high_resolution_clock::now();
  }
  ~CtranMapperTimer() = default;
  double durationMs() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               end - this->start_)
        .count();
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

class CtranMapper {
 public:
  CtranMapper(ncclComm* comm);
  ~CtranMapper();
  /* Cache and may register the given buffer and return a handle.
   * Input arguments:
   *   - buf: the local buffer to be cached, registration will happen only when
   * the eager mode is used
   *   - len: number of bytes of 'buf' to be cached and registered
   *   - forceRegist: force to register the buffer even if set lazy registration
   * Output arguments:
   *   - hdl: a handle object return by this function for future reference
   */
  ncclResult_t regMem(
      const void* buf,
      std::size_t len,
      void** hdl,
      bool forceRegist = false);
  /* Deregister and remove the handle in the registration cache.
   * Input arguments:
   *   - hdl: a handle object returned previous when registring the buffer
   */
  ncclResult_t deregMem(void* hdl);
  /* Get the handle of the given buffer if it is cached.
   * Input arguments:
   *   - buf: the local buffer to be searched in the cache
   *   - len: number of bytes of 'buf' to be searched
   * Output arguments:
   *   - hdl: a handle object if the buffer is found in the cache
   *   - dynamicRegist: whether or not this buffer is dynamically cached and
   * registered
   */
  ncclResult_t searchRegHandle(
      const void* buf,
      std::size_t len,
      void** hdl,
      bool* dynamicRegist);

  /* Post a copy op and return a reqest object.
   * Input arguments:
   *   - dbuf: destination buffer to copy the data to
   *   - sbuf: source buffer to copy the data from
   *   - len: number of bytes to copy
   * Output arguments:
   *   - req: a request object to track the progress of the copy
   */
  ncclResult_t icopy(
      void* dbuf,
      const void* sbuf,
      std::size_t len,
      CtranMapperRequest** req);
  /* Post a copy op and return a reqest object.
   * Input arguments:
   *   - dbuf: destination buffer to copy the data to
   *   - sbuf: source buffer to copy the data from
   *   - len: number of bytes to copy
   *   - stream: the CUDA stream to execute the copy on
   * Output arguments:
   *   - req: a request object to track the progress of the copy
   */
  ncclResult_t icopy(
      void* dbuf,
      const void* sbuf,
      std::size_t len,
      cudaStream_t stream,
      CtranMapperRequest** req);


  /* Post a send control op to associated backend.
   * Input arguments:
   *   - buf: the local buffer to be remotely accessed by future iput from the
   * remote peer
   *   - hdl: the handle of the buffer
   *   - rank: the rank of the remote peer in the current communicator
   * Output arguments:
   *   - req: the request object to track the progress of the control msg
   */
  ncclResult_t
  isendCtrl(void* buf, void* hdl, int rank, CtranMapperRequest** req);
  /* Post a receive control op to associated backend.
   * Input arguments:
   *   - buf: the buffer to receive the control message. It is often a buffer to
   * hold the virtual address of the remote buffer that will be accessed by
   * iput.
   *   - key: the remoteAccessKey of the remote buffer that will be updated by
   * iput. Multiple keys may exist for multiple backend transports.
   *   - rank: the rank of the remote peer in the current communicator
   * Output arguments:
   *   - req: the request object to track the progress of the control msg
   */
  ncclResult_t irecvCtrl(
      void** buf,
      struct CtranMapperRemoteAccessKey* key,
      int rank,
      CtranMapperRequest** req);

  /* Post a put op to associated backend.
   * Input arguments:
   *   - sbuf: local buffer to put data from
   *   - dbuf: virtual address of the remote buffer to receive data. It is
   *           exchanged via isendCtrl|irecvCtrl called by the algorithm
   *           layer.
   *   - len: number of bytes
   *   - rank: the rank of the remote peer in the current communicator
   *   - shdl: the handle of the source buffer
   *   - remoteAccessKey: the remote access key of the remote buffer
   *   - notify: whether notify the remote peer when finished the outstanding put.
   * Output arguments:
   *   - req: the request object to track the progress of the iput
   */
  ncclResult_t iput(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      int rank,
      void* shdl,
      struct CtranMapperRemoteAccessKey remoteAccessKey,
      bool notify,
      CtranMapperRequest** req);
  /* Check the notification with a peer rank, i.e. the completion of any
   * request. This will check all backend transports. Input arguments:
   *   - rank: the rank of the peer to check the notification
   * Output arguments:
   *   - notify: whether the peer has finished the outstanding requests
   */
  ncclResult_t checkNotify(int rank, bool* notify);
  /* Waiting for the notification from a peer rank. This will wait for all
   * backend transports. Input arguments:
   *   - rank: the rank of the peer to wait for the notification
   */
  ncclResult_t waitNotify(int rank);
  /* report the Ctran profiling results
   * Input arguments:
   *   - flush: force flushing the profiling result
   */
  void reportProfiling(bool flush = false);
  void reportRegSnapshot();

  int rank;
  uint64_t commHash;
  std::vector<std::unique_ptr<CtranMapperTimestamp>> timestamps;

 protected:
  ncclResult_t progress(void);
  cudaStream_t internalStream;

 private:
  class impl;
  std::unique_ptr<impl> pimpl_;
  friend class CtranMapperRequest;
};

#endif
