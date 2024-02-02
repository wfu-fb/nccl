// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <climits>
#include <cstddef>
#include <iostream>
#include <memory>
#include "CtranIb.h"
#include "CtranIbImpl.h"
#include "comm.h"
#include "socket.h"
#include "nccl_cvars.h"
#include "tests_common.cuh"

class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    initializeMpi(0, NULL);
    // Turn off NCCL debug logging, allow user to turn on via command line
    setenv("NCCL_DEBUG", "WARN", 0);
  }
  void TearDown() override {
    finalizeMpi();
  }
  ~MPIEnvironment() override {}
};

class CtranIbTest : public ::testing::Test {
 public:
  CtranIbTest() = default;
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
  }

  void printTestDesc(const std::string& testName, const std::string& testDesc) {
    if (this->globalRank == 0) {
      std::cout << testName << " numRanks " << this->numRanks << "."
                << std::endl
                << testDesc << std::endl;
    }
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm;
};

TEST_F(CtranIbTest, NormalInitialize) {
  this->printTestDesc(
      "NormalInitialize",
      "Expect CtranIb to be initialized without internal error.");

  try {
    auto ctranIb =
        std::unique_ptr<class CtranIb>(new class CtranIb(this->comm));
  } catch (const std::bad_alloc& e) {
    printf("CtranIbTest: IB backend not enabled. Skip test\n");
  }
}

TEST_F(CtranIbTest, RegMem) {
  this->printTestDesc(
      "RegMem",
      "Expect RegMem and deregMem can be finished without internal error.");

  try {
    auto ctranIb = std::unique_ptr<class CtranIb>(new class CtranIb(comm));
    void* buf = nullptr;
    size_t len = 2048576;
    void* handle = nullptr;

    CUDACHECK_TEST(cudaMalloc(&buf, len));
    NCCLCHECK_TEST(ctranIb->regMem(buf, len, &handle));
    EXPECT_NE(handle, nullptr);

    NCCLCHECK_TEST(ctranIb->deregMem(handle));
    CUDACHECK_TEST(cudaFree(buf));
  } catch (const std::bad_alloc& e) {
    printf("CtranIbTest: IB backend not enabled. Skip test\n");
  }
}

TEST_F(CtranIbTest, SmallRegMem) {
  this->printTestDesc(
      "SmallRegMem",
      "Expect RegMem fails due to buffer size smaller than or equal to page size.");

  // Expect registration fails due to small size <= pageSize
  try {
    auto ctranIb =
        std::unique_ptr<class CtranIb>(new class CtranIb(this->comm));
    void* buf = nullptr;
    size_t len = 4096;
    void* handle = nullptr;
    ncclResult_t res;

    CUDACHECK_TEST(cudaMalloc(&buf, len));
    res = ctranIb->regMem(buf, len, &handle);
    EXPECT_EQ(res, ncclSystemError);
    EXPECT_EQ(handle, nullptr);
    CUDACHECK_TEST(cudaFree(buf));

    len = 512;
    CUDACHECK_TEST(cudaMalloc(&buf, len));
    res = ctranIb->regMem(buf, len, &handle);
    EXPECT_EQ(res, ncclSystemError);
    EXPECT_EQ(handle, nullptr);
    CUDACHECK_TEST(cudaFree(buf));

  } catch (const std::bad_alloc& e) {
    printf("CtranIbTest: IB backend not enabled. Skip test\n");
  }
}

TEST_F(CtranIbTest, CpuMemSendRecvCtrl) {
  this->printTestDesc(
      "CpuMemSendRecvCtrl",
      "Expect rank 0 can isendCtrl its local CPU buffer's address and rkey to rank 1 who calls irecvCtrl. "
      "The received rkey and remoteAddr should not be zero.");

  try {
    auto ctranIb = std::unique_ptr<class CtranIb>(new class CtranIb(comm));
    char buf[8192];
    void* remoteBuf = nullptr;
    void* handle = nullptr;
    struct CtranIbRemoteAccessKey key = {0};
    CtranIbRequest* req;
    ncclResult_t res;

    NCCLCHECK_TEST(ctranIb->regMem(buf, 8192, &handle));
    if (this->globalRank == 0) {
      NCCLCHECK_TEST(ctranIb->isendCtrl(buf, handle, 1, &req));
    } else if (this->globalRank == 1) {
      NCCLCHECK_TEST(ctranIb->irecvCtrl(&remoteBuf, &key, 0, &req));
    }

    do {
      NCCLCHECK_TEST(ctranIb->progress());
    } while (!req->isComplete());

    if (this->globalRank == 1) {
      EXPECT_NE(remoteBuf, nullptr);
      EXPECT_NE(key.rkey, 0);
    }

    NCCLCHECK_TEST(ctranIb->deregMem(handle));
    delete req;
  } catch (const std::bad_alloc& e) {
    printf("CtranIbTest: IB backend not enabled. Skip test\n");
  }
}

TEST_F(CtranIbTest, GpuMemSendRecvCtrl) {
  this->printTestDesc(
      "CpuMemSendRecvCtrl",
      "Expect rank 0 can isendCtrl its local GPU buffer's address and rkey to rank 1 who calls irecvCtrl. "
      "The received rkey and remoteAddr should not be zero.");

  try {
    auto ctranIb = std::unique_ptr<class CtranIb>(new class CtranIb(comm));
    void* buf;
    void* remoteBuf = nullptr;
    void* handle = nullptr;
    struct CtranIbRemoteAccessKey key = {0};
    CtranIbRequest* req;
    ncclResult_t res;

    CUDACHECK_TEST(cudaMalloc(&buf, 8192));
    NCCLCHECK_TEST(ctranIb->regMem(buf, 8192, &handle));
    if (this->globalRank == 0) {
      NCCLCHECK_TEST(ctranIb->isendCtrl(buf, handle, 1, &req));
    } else if (this->globalRank == 1) {
      NCCLCHECK_TEST(ctranIb->irecvCtrl(&remoteBuf, &key, 0, &req));
    }

    do {
      NCCLCHECK_TEST(ctranIb->progress());
    } while (!req->isComplete());

    if (this->globalRank == 1) {
      EXPECT_NE(remoteBuf, nullptr);
      EXPECT_NE(key.rkey, 0);
    }

    NCCLCHECK_TEST(ctranIb->deregMem(handle));
    CUDACHECK_TEST(cudaFree(buf));
    delete req;
  } catch (const std::bad_alloc& e) {
    printf("CtranIbTest: IB backend not enabled. Skip test\n");
  }
}

TEST_F(CtranIbTest, CpuMemPutNotify) {
  this->printTestDesc(
      "CpuMemPutNotify",
      "Expect rank 0 can put data from its local CPU data to rank 1 who waits on notify. "
      "The received data should be equal to send data on rank 0.");

#undef BUF_COUNT
#define BUF_COUNT 8192
  try {
    auto ctranIb = std::unique_ptr<class CtranIb>(new class CtranIb(comm));
    int buf[BUF_COUNT];
    void* remoteBuf = nullptr;
    void* handle = nullptr;
    struct CtranIbRemoteAccessKey key = {0};
    CtranIbRequest *ctrlReq, *putReq;
    const int sendVal = 99, recvVal = -1;
    const int sendRank = 0, recvRank = 1;

    // fill the buffer with different values
    for (int i = 0; i < BUF_COUNT; i++) {
      buf[i] = this->globalRank == sendRank ? sendVal : recvVal;
    }

    // Register
    NCCLCHECK_TEST(ctranIb->regMem(buf, BUF_COUNT * sizeof(int), &handle));

    // Receiver sends the remoteAddr and rkey to sender
    if (this->globalRank == recvRank) {
      NCCLCHECK_TEST(ctranIb->isendCtrl(buf, handle, 0, &ctrlReq));
    } else if (this->globalRank == 0) {
      NCCLCHECK_TEST(ctranIb->irecvCtrl(&remoteBuf, &key, 1, &ctrlReq));
    }

    // Waits control message to be received
    do {
      NCCLCHECK_TEST(ctranIb->progress());
    } while (!ctrlReq->isComplete());

    // Sender puts the data to rank 1
    if (this->globalRank == sendRank) {
      ctranIb->iput(
          buf,
          remoteBuf,
          BUF_COUNT * sizeof(int),
          1,
          handle,
          key,
          true,
          &putReq);

      // waits for put to finish
      do {
        NCCLCHECK_TEST(ctranIb->progress());
      } while (!putReq->isComplete());
    } else {
      // Receiver waits notify and check data
      NCCLCHECK_TEST(ctranIb->waitNotify(sendRank));

      for (int i = 0; i < BUF_COUNT; i++) {
        EXPECT_EQ(buf[i], sendVal);
      }
    }

    NCCLCHECK_TEST(ctranIb->deregMem(handle));
    delete ctrlReq, putReq;
  } catch (const std::bad_alloc& e) {
    printf("CtranIbTest: IB backend not enabled. Skip test\n");
  }
}

TEST_F(CtranIbTest, GpuMemPutNotify) {
  this->printTestDesc(
      "GpuMemPutNotify",
      "Expect rank 0 can put data from its local GPU data to rank 1 who waits on notify. "
      "The received data should be equal to send data on rank 0.");

#undef BUF_COUNT
#define BUF_COUNT 8192
  try {
    auto ctranIb = std::unique_ptr<class CtranIb>(new class CtranIb(comm));
    int* buf;
    int hostBuf[BUF_COUNT];
    void* remoteBuf = nullptr;
    void* handle = nullptr;
    struct CtranIbRemoteAccessKey key = {0};
    CtranIbRequest *ctrlReq, *putReq;
    const int sendVal = 99, recvVal = -1;
    const int sendRank = 0, recvRank = 1;

    CUDACHECK_TEST(cudaMalloc(&buf, BUF_COUNT * sizeof(int)));

    // fill the buffer with different values and copy to GPU
    for (int i = 0; i < BUF_COUNT; i++) {
      hostBuf[i] = this->globalRank == sendRank ? sendVal : recvVal;
    }

    CUDACHECK_TEST(cudaMemcpy(
        buf, hostBuf, BUF_COUNT * sizeof(int), cudaMemcpyHostToDevice));

    // Register
    NCCLCHECK_TEST(ctranIb->regMem(buf, BUF_COUNT * sizeof(int), &handle));

    // Receiver sends the remoteAddr and rkey to sender
    if (this->globalRank == recvRank) {
      NCCLCHECK_TEST(ctranIb->isendCtrl(buf, handle, 0, &ctrlReq));
    } else if (this->globalRank == 0) {
      NCCLCHECK_TEST(ctranIb->irecvCtrl(&remoteBuf, &key, 1, &ctrlReq));
    }

    // Waits control message to be received
    do {
      NCCLCHECK_TEST(ctranIb->progress());
    } while (!ctrlReq->isComplete());

    // Sender puts the data to rank 1
    if (this->globalRank == sendRank) {
      ctranIb->iput(
          buf,
          remoteBuf,
          BUF_COUNT * sizeof(int),
          1,
          handle,
          key,
          true,
          &putReq);

      // waits for put to finish
      do {
        NCCLCHECK_TEST(ctranIb->progress());
      } while (!putReq->isComplete());
    } else {
      // Receiver waits notify and check data
      NCCLCHECK_TEST(ctranIb->waitNotify(sendRank));

      CUDACHECK_TEST(cudaMemcpy(
          hostBuf, buf, BUF_COUNT * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < BUF_COUNT; i++) {
        EXPECT_EQ(hostBuf[i], sendVal);
      }
    }

    NCCLCHECK_TEST(ctranIb->deregMem(handle));
    CUDACHECK_TEST(cudaFree(buf));
    delete ctrlReq, putReq;
  } catch (const std::bad_alloc& e) {
    printf("CtranIbTest: IB backend not enabled. Skip test\n");
  }
}

TEST_F(CtranIbTest, MultiPutTrafficProfiler) {
  this->printTestDesc(
      "MultiPutTrafficProfiler",
      "Expect rank 0 puts data from its local GPU data to other ranks and "
      "the traffic profiling can catch exact bytes as expected per device and per QP.");

  setenv("NCCL_CTRAN_IB_TRAFFIC_PROFILNG", "true", 1);
  ncclCvarInit();

#undef BUF_COUNT
#define BUF_COUNT 8192
  try {
    auto ctranIb = std::unique_ptr<class CtranIb>(new class CtranIb(comm));
    int* buf;
    void* handle = nullptr;
    std::vector<struct CtranIbRemoteAccessKey> remoteKeys(this->numRanks);
    std::vector<void*> remoteBufs(this->numRanks);
    std::unique_ptr<CtranIbRequest> ctrlSReq;
    std::vector<std::unique_ptr<CtranIbRequest>> ctrlRReqs(this->numRanks);
    std::vector<std::unique_ptr<CtranIbRequest>> putReqs(this->numRanks);
    CtranIbRequest *req = nullptr;
    const int rootRank = 0;

    CUDACHECK_TEST(cudaSetDevice(this->globalRank));

    // Allocate and register buffer
    CUDACHECK_TEST(cudaMalloc(&buf, BUF_COUNT * sizeof(int)));
    NCCLCHECK_TEST(ctranIb->regMem(buf, BUF_COUNT * sizeof(int), &handle));

    // rootRank receives remoteAddr from all ranks
    if (this->globalRank == rootRank) {
      for (int i = 0; i < this->numRanks; i++) {
        // skip rootRank itself
        if (i == rootRank) {
          continue;
        }

        NCCLCHECK_TEST(ctranIb->irecvCtrl(&remoteBufs[i], &remoteKeys[i], i, &req));
        ctrlRReqs[i] = std::unique_ptr<CtranIbRequest>(req);
      }
    } else {
      // other rank sends the remoteAddr and rkey to rootRank
      NCCLCHECK_TEST(ctranIb->isendCtrl(buf, handle, rootRank, &req));
      ctrlSReq = std::unique_ptr<CtranIbRequest>(req);
    }

    // rootRank puts data to other ranks
    if (this->globalRank == rootRank) {
      int nPendingPuts = this->numRanks - 1;
      while(nPendingPuts > 0) {
        for (int i = 0; i < this->numRanks; i++) {
          // skip rootRank itself
          if (i == rootRank) {
            continue;
          }

          // wait control messages to be received from this sender
          // or skip if has already put
          if (!ctrlRReqs[i]->isComplete() || putReqs[i]) {
            NCCLCHECK_TEST(ctranIb->progress());
            continue;
          }

          // put data to sender
          ctranIb->iput(
              buf,
              remoteBufs[i],
              BUF_COUNT * sizeof(int),
              i,
              handle,
              remoteKeys[i],
              true,
              &req);
          putReqs[i] = std::unique_ptr<CtranIbRequest>(req);
          nPendingPuts--;
        }
      }

      // waits for all put to finish
      int nCompletedPuts = this->numRanks - 1;
      do {
        NCCLCHECK_TEST(ctranIb->progress());
        for (int i = 0; i < this->numRanks; i++) {
          // skip rootRank itself
          if (i == rootRank) {
            continue;
          }

          if (putReqs[i]->isComplete()) {
            nCompletedPuts--;
          }
        }
      } while (nCompletedPuts > 0);
    } else {
      // Other rank ensures send control messages has completed
      while (!ctrlSReq->isComplete()) {
        NCCLCHECK_TEST(ctranIb->progress());
      }

      // Other rank waits notify to safely free buffer
      NCCLCHECK_TEST(ctranIb->waitNotify(rootRank));
    }

    // Rank 0 checks traffic snapshot
    if (this->globalRank == rootRank) {
      CtranIbSingleton& s = CtranIbSingleton::getInstance();
      auto devSnapshot = s.getDeviceTrafficSnapshot();
      for (auto& it : devSnapshot) {
        // all data sent to other ranks should go through the same device
        EXPECT_EQ(it.second, (this->numRanks - 1) * BUF_COUNT * sizeof(int));
      }

      auto qpSnapshot = s.getQpTrafficSnapshot();
      for (auto& it : qpSnapshot) {
        // data sent to each rank should go through different QP
        EXPECT_EQ(it.second, BUF_COUNT * sizeof(int));
      }
    }

    // Free resources
    NCCLCHECK_TEST(ctranIb->deregMem(handle));
    CUDACHECK_TEST(cudaFree(buf));

  } catch (const std::bad_alloc& e) {
    printf("CtranIbTest: IB backend not enabled. Skip test\n");
  }

  unsetenv("NCCL_CTRAN_IB_TRAFFIC_PROFILNG");
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
