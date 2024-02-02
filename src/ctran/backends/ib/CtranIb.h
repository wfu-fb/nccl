// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_IB_H_
#define CTRAN_IB_H_

#include "nccl.h"
#include <memory>

struct CtranIbRemoteAccessKey {
  uint32_t rkey;
};

class CtranIb;
struct ncclComm;

/**
 * Class of request to track progress of a isendCtrl, irecvCtrl, or iput IB
 * operation.
 */
class CtranIbRequest {
 public:
  CtranIbRequest(){};
  ~CtranIbRequest(){};

  // Mark the number of expected references associated with the request (e.g.,
  // multiple references if a data is internally chunked to multiple IB packets
  // and issued via multiple QPs). Default set refCount to 1 when creating the
  // request.
  void setRefCount(int refCount);

  // Mark completion of a reference
  void complete();

  // Return true if all references have been completed. Otherwise false.
  bool isComplete();

 private:
  enum {
    INCOMPLETE,
    COMPLETE,
  } state_{INCOMPLETE};
  int refCount_{1};
};

/**
* CtranIB class to be used by algorithms and ctranMapper.
*/
class CtranIb {
public:
 // Creates local IB resources for a given communicator including obtaining the
 // singleton PD and context, and creating a per-communicator Completion Queue
 // (CQ). It also launches a listen thread to accept remote connection. The
 // remote connection will happen when the remote peer issues the first message
 // to the local rank.
 // Input arguments:
 //   - comm: the NCCL communicator
 CtranIb(ncclComm* comm);
 ~CtranIb();

 // Register memory to be used for IB operations.
 // Input arguments:
 //   - buf: the local buffer to be registered to network for direct RDMA access
 //   - len: the length of the local buffer
 // Output arguments:
 //   - ibRegElem: the ibRegElem of the local buffer that stores the
 //                registration handle.
 ncclResult_t regMem(const void* buf, std::size_t len, void** ibRegElem);

 // Deregister memory to be used for IB operations.
 // Input arguments:
 //   - ibRegElem: the ibRegElem of the local buffer that stores the
 //                registration handle.
 ncclResult_t deregMem(void* ibRegElem);

 // Progress the per-communicator CQ.
 ncclResult_t progress(void);

 // Send control message over the established IB
 // connection.
 // Input arguments:
 //   - buf: the local buffer to be remotely accessed by coming
 //          iput from the remote peer
 //   - ibRegElem: the ibRegElem of the local buffer
 //   - rank: the rank of the remote peer in the current communicator
 // Output arguments:
 //   - req: the request object to track the progress of the send
 ncclResult_t
 isendCtrl(void* buf, void* ibRegElem, int rank, CtranIbRequest** req);

 // Receive control message over the established IB
 // connection.
 // Input arguments:
 //   - buf: the buffer to receive the control message. It is often a buffer
 //          to hold the virtual address of the remote buffer that will be
 //          accessed by iput.
 //   - key: the remoteAccessKey of the remote buffer that will be updated by
 //          iput.
 //   - rank: the rank of the remote peer in the current communicator
 // Output arguments:
 //   - req: the request object to track the progress of the receive
 ncclResult_t irecvCtrl(
     void** buf,
     struct CtranIbRemoteAccessKey* key,
     int rank,
     CtranIbRequest** req);

 // RDMA put data from local sbuf to a dbuf in remote rank over
 // the established IB connection.
 // Input arguments:
 //   - sbuf: local buffer to put data from
 //   - dbuf: virtual address of the remote buffer to receive data. It is
 //           exchanged via isendCtrl|irecvCtrl called by the algorithm
 //           layer.
 //   - len: length of data
 //   - rank: the rank of the remote peer in the current communicator
 //   - ibRegElem: the ibRegElem of the local sbuf
 //   - remoteAccessKey: the remoteAccessKey of dbuf. It is exchanged via
 //                      isendCtrl|irecvCtrl called by the algorithm layer.
 //   - notify: whether to notify the remote peer when the RDMA PUT has finished
 //             and data has arrived in the remote dbuf.
 // Output arguments:
 //   - req: the request object to track the progress of the iput
 ncclResult_t iput(
     const void* sbuf,
     void* dbuf,
     std::size_t len,
     int rank,
     void* ibRegElem,
     struct CtranIbRemoteAccessKey remoteAccessKey,
     bool notify,
     CtranIbRequest** req);

 // Check whether the remote rank has finished the outstanding iput
 // Input arguments:
 //   - rank: the rank of the remote peer in the current communicator that has
 //           issued iput to the local rank.
 // Output arguments:
 //   - notify: whether the remote peer has finished the outstanding iput.
 ncclResult_t checkNotify(int rank, bool* notify);

 // Wait until the remote rank has finished the outstanding iput
 // Input arguments:
 //   - rank: the rank of the remote peer in the current communicator that has
 //           issued iput to the local rank.
 ncclResult_t waitNotify(int rank);

 std::string getIbDevName();

 int getIbDevPort();

private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
  friend class CtranIbRequest;
};

#endif
