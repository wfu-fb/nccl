// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ctranAlgos.h"
#include "comm.h"
#include <deque>

thread_local std::deque<struct collOp *> ctranOpGroup;

static ncclResult_t sendRecvImpl(std::vector<std::unique_ptr<struct collOp>> opGroup) {
  ncclResult_t res = ncclSuccess;
  std::vector<struct collOp *> sendOpGroup;
  std::vector<struct collOp *> recvOpGroup;

  for (auto& op : opGroup) {
    if (op->type == collOp::opType::SEND) {
      sendOpGroup.push_back(op.get());
    } else {
      recvOpGroup.push_back(op.get());
    }
  }

  void *sendMemHdl[sendOpGroup.size()];
  void *remoteRecvBuff[sendOpGroup.size()];
  struct ctranMapperRemoteAccessKey remoteAccessKey[sendOpGroup.size()];
  ctranMapperRequest *sendCtrlReqs[sendOpGroup.size()];
  ctranMapperRequest *putReqs[sendOpGroup.size()];
  bool putIssued[sendOpGroup.size()];

  void *recvMemHdl[recvOpGroup.size()];
  ctranMapperRequest *recvCtrlReqs[recvOpGroup.size()];
  int recvPeerRanks[recvOpGroup.size()];

  ncclComm_t comm = opGroup.front()->comm;
  ctranMapper *mapper = comm->ctranMapper;

  std::vector<void *> tmpRegHdls;

  /* issue control messages for send operations */
  for (auto i = 0; i < sendOpGroup.size(); i++) {
    auto op = sendOpGroup[i];
    size_t sendSize = op->send.count * ncclTypeSize(op->send.datatype);

    NCCLCHECKGOTO(mapper->searchRegHandle(op->send.sendbuff, sendSize, &sendMemHdl[i]),
        res, exit);
    if (sendMemHdl[i] == nullptr) {
      NCCLCHECKGOTO(mapper->regMem(op->send.sendbuff, sendSize, &sendMemHdl[i]), res, exit);
      tmpRegHdls.push_back(sendMemHdl[i]);
    }

    NCCLCHECKGOTO(mapper->irecvCtrl(&remoteRecvBuff[i], &remoteAccessKey[i],
          op->send.peerRank, &sendCtrlReqs[i]), res, exit);
    putIssued[i] = false;
  }

  /* issue control messages for recv operations */
  for (auto i = 0; i < recvOpGroup.size(); i++) {
    auto op = recvOpGroup[i];
    size_t recvSize = op->recv.count * ncclTypeSize(op->recv.datatype);

    NCCLCHECKGOTO(mapper->searchRegHandle(op->recv.recvbuff, recvSize, &recvMemHdl[i]),
        res, exit);
    if (recvMemHdl[i] == nullptr) {
      NCCLCHECKGOTO(mapper->regMem(op->recv.recvbuff, recvSize, &recvMemHdl[i]), res, exit);
      tmpRegHdls.push_back(recvMemHdl[i]);
    }

    ctranMapperRequest *req;
    NCCLCHECKGOTO(mapper->isendCtrl(op->recv.recvbuff, recvMemHdl[i], op->recv.peerRank,
          &recvCtrlReqs[i]), res, exit);
    recvPeerRanks[i] = op->recv.peerRank;
  }

  /* as we recv control msgs, issue PUT operations */
  while (1) {
    bool pendingOps = false;

    for (auto i = 0; i < sendOpGroup.size(); i++) {
      if (putIssued[i] == true) {
        continue;
      } else {
        auto op = sendOpGroup[i];
        size_t sendSize = op->send.count * ncclTypeSize(op->send.datatype);
        bool isComplete;

        NCCLCHECKGOTO(sendCtrlReqs[i]->test(&isComplete), res, exit);
        if (isComplete) {
          NCCLCHECKGOTO(mapper->iput(op->send.sendbuff, remoteRecvBuff[i], sendSize, op->send.peerRank,
                sendMemHdl[i], remoteAccessKey[i], true, &putReqs[i]), res, exit);
          putIssued[i] = true;
        } else {
          pendingOps = true;
        }
      }
    }

    if (pendingOps == false) {
      break;
    }
  }

  /* wait for all PUT messages to complete */
  for (auto i = 0; i < sendOpGroup.size(); i++) {
    NCCLCHECKGOTO(putReqs[i]->wait(), res, exit);
  }

  /* wait for all control messages and notifications to complete */
  for (auto i = 0; i < recvOpGroup.size(); i++) {
    NCCLCHECKGOTO(recvCtrlReqs[i]->wait(), res, exit);
    NCCLCHECKGOTO(mapper->waitNotify(recvPeerRanks[i]), res, exit);
  }

  /* deregister temporary registrations */
  for (auto hdl : tmpRegHdls) {
    NCCLCHECKGOTO(mapper->deregMem(hdl), res, exit);
  }

exit:
  return res;
}

ncclResult_t ctranSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  struct collOp *op;

  op = new struct collOp;
  op->type = collOp::opType::SEND;
  op->comm = comm;
  op->stream = stream;
  op->send.sendbuff = sendbuff;
  op->send.count = count;
  op->send.datatype = datatype;
  op->send.peerRank = peer;

  ctranOpGroup.push_back(op);

  return res;
}

ncclResult_t ctranRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t res = ncclSuccess;
  struct collOp *op;

  op = new struct collOp;
  op->type = collOp::opType::RECV;
  op->comm = comm;
  op->stream = stream;
  op->recv.recvbuff = recvbuff;
  op->recv.count = count;
  op->recv.datatype = datatype;
  op->recv.peerRank = peer;

  ctranOpGroup.push_back(op);

  return res;
}

ncclResult_t ctranGroupEndHook(void) {
  ncclResult_t res = ncclSuccess;
  ncclComm_t comm;
  cudaStream_t stream;

  while (1) {
    std::vector<std::unique_ptr<struct collOp>> toSubmit;
    std::deque<struct collOp *> pending;
    bool hasSend = false;
    bool hasRecv = false;

    if (ctranOpGroup.empty()) {
      break;
    }

    comm = ctranOpGroup.front()->comm;
    stream = ctranOpGroup.front()->stream;
    while (!ctranOpGroup.empty()) {
      struct collOp *op = ctranOpGroup.front();
      ctranOpGroup.pop_front();

      if (op->comm == comm && op->stream == stream) {
        toSubmit.push_back(std::unique_ptr<struct collOp>(op));
        if (op->type == collOp::opType::SEND) {
          hasSend = true;
        } else if (op->type == collOp::opType::RECV) {
          hasRecv = true;
        }
      } else {
        pending.push_back(op);
      }
    }

    if (hasSend && hasRecv) {
      NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(toSubmit), sendRecvImpl,
            reinterpret_cast<void *>(ncclKernelSendRecv)), res, exit);
    } else if (hasSend) {
      NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(toSubmit), sendRecvImpl,
            reinterpret_cast<void *>(ncclKernelSend)), res, exit);
    } else if (hasRecv) {
      NCCLCHECKGOTO(comm->ctranGpe->submit(std::move(toSubmit), sendRecvImpl,
            reinterpret_cast<void *>(ncclKernelRecv)), res, exit);
    }

    toSubmit.clear();
    ctranOpGroup = std::move(pending);
  }

exit:
  return res;
}
