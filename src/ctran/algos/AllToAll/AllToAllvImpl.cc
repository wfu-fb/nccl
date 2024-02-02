// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "AllToAllvImpl.h"
#include <nccl.h>
#include <cstddef>
#include <vector>
#include "Ctran.h"
#include "comm.h"

static inline ncclResult_t regIsendCtrl(
    ncclComm_t& comm,
    int peer,
    void* recvBuff,
    size_t recvBytes,
    std::vector<void*>& recvMemHdl,
    std::vector<void*>& tmpRegHdls,
    std::vector<std::unique_ptr<CtranMapperRequest>>& reqs) {
  bool localReg = false;
  CtranMapperRequest* req = nullptr;
  NCCLCHECK(comm->ctran->mapper->searchRegHandle(
      recvBuff, recvBytes, &recvMemHdl[peer], &localReg));
  if (localReg) {
    tmpRegHdls.push_back(recvMemHdl[peer]);
  }
  NCCLCHECK(
      comm->ctran->mapper->isendCtrl(recvBuff, recvMemHdl[peer], peer, &req));
  reqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
  return ncclSuccess;
}

static inline ncclResult_t regIrecvCtrl(
    ncclComm_t& comm,
    int peer,
    const void* sendBuff,
    size_t sendBytes,
    std::vector<void*>& sendMemHdl,
    std::vector<void*>& tmpRegHdls,
    std::vector<void*>& remoteRecvBuffs,
    std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
    std::vector<std::unique_ptr<CtranMapperRequest>>& reqs) {
  CtranMapperRequest* req = nullptr;
  bool localReg = false;
  NCCLCHECK(comm->ctran->mapper->searchRegHandle(
      sendBuff, sendBytes, &sendMemHdl[peer], &localReg));
  if (localReg) {
    tmpRegHdls.push_back(sendMemHdl[peer]);
  }

  NCCLCHECK(comm->ctran->mapper->irecvCtrl(
      &remoteRecvBuffs[peer], &remoteAccessKeys[peer], peer, &req));
  reqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
  return ncclSuccess;
}

ncclResult_t ctranAllToAllvIbImpl(
    const void* sendbuff,
    std::vector<size_t>& sendCounts,
    std::vector<size_t>& sDispls,
    void* recvbuff,
    std::vector<size_t>& recvCounts,
    std::vector<size_t>& rDispls,
    ncclDataType_t datatype,
    ncclComm_t comm,
    std::unique_ptr<CtranMapperTimestamp> timestamp) {
  ncclResult_t res = ncclSuccess;

  std::vector<const void*> sendBuffs(comm->nRanks);
  std::vector<void*> recvBuffs(comm->nRanks);
  std::vector<void*> remoteRecvBuffs(comm->nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(comm->nRanks);

  std::vector<std::unique_ptr<CtranMapperRequest>> ibSendCtrlReqs,
      ibRecvCtrlReqs;
  std::vector<std::unique_ptr<CtranMapperRequest>> ibPutReqs;

  std::vector<void*> sendMemHdl(comm->nRanks);
  std::vector<void*> recvMemHdl(comm->nRanks);
  std::vector<void*> tmpRegHdls;
  CtranMapperRequest* req = nullptr;

  std::vector<int> ibRecvPeers, ibSendPeers;

  // Prepare buffers shifted with displacement, and set ctrl/put/notify
  // schedules. Try to schedule ctrl message and put sequence as rank i start
  // sending to rank i+1 to avoid congestion in potential all-to-one case.
  // Specified in putPeers, sendCtrlPeers.
  for (int i = 0; i < comm->nRanks; i++) {
    int peer = (comm->rank + i) % comm->nRanks;
    if (sendCounts[peer]) {
      sendBuffs[peer] = static_cast<const char*>(sendbuff) +
          sDispls[peer] * ncclTypeSize(datatype);
      ibSendPeers.push_back(peer);
    }
    if (recvCounts[peer]) {
      recvBuffs[peer] =
          static_cast<char*>(recvbuff) + rDispls[peer] * ncclTypeSize(datatype);
      ibRecvPeers.push_back(peer);
    }
  }

  // schedule IB ctrl messages
  ibSendCtrlReqs.reserve(ibRecvPeers.size());
  ibRecvCtrlReqs.reserve(ibSendPeers.size());
  for (auto peer : ibRecvPeers) {
    NCCLCHECKGOTO(
        regIsendCtrl(
            comm,
            peer,
            recvBuffs[peer],
            recvCounts[peer] * ncclTypeSize(datatype),
            recvMemHdl,
            tmpRegHdls,
            ibSendCtrlReqs),
        res,
        exit);
  }

  for (auto peer : ibSendPeers) {
    NCCLCHECKGOTO(
        regIrecvCtrl(
            comm,
            peer,
            sendBuffs[peer],
            sendCounts[peer] * ncclTypeSize(datatype),
            sendMemHdl,
            tmpRegHdls,
            remoteRecvBuffs,
            remoteAccessKeys,
            ibRecvCtrlReqs),
        res,
        exit);
  }

  // issue network puts:
  // - Sender puts data for peers, whenever received the remote recvbuff handle
  // - Exit until all peers' put have been issued (putPeers becomes empty)
  ibPutReqs.reserve(ibSendPeers.size());
  while (!ibRecvCtrlReqs.empty()) {
    auto it = ibRecvCtrlReqs.begin();
    while (it != ibRecvCtrlReqs.end()) {
      auto& recvCtrlReq = *it;
      int peer = recvCtrlReq->peer;

      bool completed = false;
      NCCLCHECKGOTO(recvCtrlReq->test(&completed), res, exit);
      if (completed) {
        timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(peer));
        NCCLCHECKGOTO(
            comm->ctran->mapper->iput(
                sendBuffs[peer],
                remoteRecvBuffs[peer],
                sendCounts[peer] * ncclTypeSize(datatype),
                peer,
                sendMemHdl[peer],
                remoteAccessKeys[peer],
                true, /* notify receiver when completes */
                &req),
            res,
            exit);
        ibPutReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
        timestamp->putIssued.push_back(CtranMapperTimestampPoint(peer));
        it = ibRecvCtrlReqs.erase(it);
      } else {
        it++;
      }
    }
  }

  // Wait for all puts to complete
  while (!ibPutReqs.empty()) {
    NCCLCHECKGOTO(
        comm->ctran->mapper->testSomeRequests(
            ibPutReqs, timestamp->putComplete),
        res,
        exit);
  }
  // Wait for all receives (i.e., remote IB puts) to complete
  while (!ibRecvPeers.empty()) {
    NCCLCHECKGOTO(comm->ctran->mapper->checkSomeNotify(ibRecvPeers), res, exit);
  }

  // Skip sendCtrl check since remote put completion should indicate completion
  // of corresponding recvCtrls

  comm->ctran->mapper->timestamps.emplace_back(std::move(timestamp));
  comm->ctran->mapper->reportProfiling();

exit:
  /* deregister temporary registrations */
  // FIXME: let GPE kernel to finish then deregister to avoid race condition on
  // cuda context
  for (auto& hdl : tmpRegHdls) {
    NCCLCHECK(comm->ctran->mapper->deregMem(hdl));
  }
  return res;
}
