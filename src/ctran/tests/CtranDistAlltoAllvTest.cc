// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <new>
#include "Ctran.h"
#include "checks.h"
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

class ctranAllToAllvTest : public ::testing::Test {
 public:
  ctranAllToAllvTest() = default;

  void generateDistRandomExpValue() {
    if (globalRank == 0) {
      expectedVal = rand();
    }
    MPI_Bcast(&expectedVal, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  void generateFixedCountsDisps(size_t count) {
    // each send/recv are with the same count and displacement
    int stride = count * 2;
    sendTotalCount = stride * numRanks;
    recvTotalCount = stride * numRanks;
    for (int i = 0; i < numRanks; ++i) {
      sendCounts[i] = count;
      sendDisps[i] = stride * i;
      recvCounts[i] = count;
      recvDisps[i] = stride * i;
    }
  }

  void generateDistRandomCountsDisps() {
    std::vector<MPI_Request> reqs(numRanks * 2, MPI_REQUEST_NULL);

    // assign random send count for each peer
    srand(time(NULL) + globalRank);

    sendTotalCount = 0;
    for (int i = 0; i < numRanks; ++i) {
      sendCounts[i] = (rand() % 10) * getpagesize(); // always page aligned size
      sendDisps[i] = sendTotalCount;
      sendTotalCount += sendCounts[i];
      // exchange send count to receiver side
      MPI_Isend(&sendCounts[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &reqs[i]);
      MPI_Irecv(
          &recvCounts[i],
          1,
          MPI_INT,
          i,
          0,
          MPI_COMM_WORLD,
          &reqs[numRanks + i]);
    }
    MPI_Waitall(numRanks * 2, reqs.data(), MPI_STATUSES_IGNORE);

    // updates recvDisp based on received counts from sender
    recvTotalCount = 0;
    for (int i = 0; i < numRanks; ++i) {
      recvDisps[i] = recvTotalCount;
      recvTotalCount += recvCounts[i];
    }
  }

  void* createDataBuf(size_t nbytes, void** handle) {
    void* buf = nullptr;
    // Allocate data buffer, and assign different value for each send chunk
    CUDACHECK_TEST(cudaMalloc(&buf, nbytes));
    if (buf && handle) {
      NCCLCHECK_TEST(ncclCommRegister(comm, buf, nbytes, handle));
    }
    return buf;
  }

  void releaseDataBuf(void* buf, void* handle) {
    if (handle) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, handle));
    }
    CUDACHECK_TEST(cudaFree(buf));
  }

  template <typename T>
  void assignChunkValue(T* buf, size_t count, T val) {
    std::vector<T> expectedVals(count, val);
    CUDACHECK_TEST(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }

  template <typename T>
  int checkChunkValue(T* buf, size_t count, T val) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print first 10 failing
    // location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != val) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d] = %d, expectedVal = %d\n",
              globalRank,
              i,
              observedVals[i],
              val);
        }
        errs++;
      }
    }
    return errs;
  }

  bool checkTestPrerequisite() {
    EXPECT_NE(nullptr, comm);
    EXPECT_NE(nullptr, comm->ctran);
    if (!ctranInitialized(comm)) {
      if (globalRank == 0) {
        printf("Skip test because ctran is not initialized\n");
      }
      return false;
    }
    return true;
  }

  void SetUp() override {
    std::tie(localRank, globalRank, numRanks) = getMpiInfo();
    comm = createNcclComm(globalRank, numRanks, localRank);

    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream));

    // Allocate enough space for arguments, value assignment set in each test
    sendBuf = nullptr;
    recvBuf = nullptr;
    sendHdl = nullptr;
    recvHdl = nullptr;
    sendCounts.resize(numRanks, 0);
    recvCounts.resize(numRanks, 0);
    sendDisps.resize(numRanks, 0);
    recvDisps.resize(numRanks, 0);
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    NCCLCHECK_TEST(ncclCommDestroy(comm));
  }

  void run() {
    // Assign different value for each send chunk
    for (int i = 0; i < numRanks; ++i) {
      assignChunkValue<int>(
          sendBuf + sendDisps[i],
          sendCounts[i],
          expectedVal + globalRank * 100 + i + 1);
    }

    if (checkTestPrerequisite()) {
      // Run communication
      for (int x = 0; x < 1; x++) {
        auto res = ctranAllToAllv(
            sendBuf,
            sendCounts.data(),
            sendDisps.data(),
            recvBuf,
            recvCounts.data(),
            recvDisps.data(),
            ncclInt,
            comm,
            stream);
        ASSERT_EQ(res, ncclSuccess);
      }
      CUDACHECK_TEST(cudaStreamSynchronize(stream));

      // Check each received chunk
      for (int i = 0; i < numRanks; ++i) {
        int errs = checkChunkValue<int>(
            recvBuf + recvDisps[i],
            recvCounts[i],
            expectedVal + i * 100 + globalRank + 1);
        EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << i
                           << " at " << recvBuf + recvDisps[i] << " with "
                           << errs << " errors";
      }
    }
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};

  cudaStream_t stream{0};
  ncclComm_t comm{nullptr};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  std::vector<size_t> sendCounts;
  std::vector<size_t> recvCounts;
  std::vector<size_t> sendDisps;
  std::vector<size_t> recvDisps;
  size_t sendTotalCount{0};
  size_t recvTotalCount{0};
  void* sendHdl{nullptr};
  void* recvHdl{nullptr};
  int expectedVal{0};
};

TEST_F(ctranAllToAllvTest, AllToAllv) {
  generateDistRandomCountsDisps();
  generateDistRandomExpValue();

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);

  run();

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

TEST_F(ctranAllToAllvTest, AllToAll) {
  generateFixedCountsDisps(1024 * 1024UL);
  generateDistRandomExpValue();

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);

  run();

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

TEST_F(ctranAllToAllvTest, ZeroByteAllToAllv) {
  generateFixedCountsDisps(0);

  // reassign non-zero total buffer sizes
  sendTotalCount = 1048576;
  recvTotalCount = 1048576;

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);

  // Reset buffers' value
  assignChunkValue(sendBuf, sendTotalCount, globalRank);
  assignChunkValue(recvBuf, recvTotalCount, -1);

  run();

  // Check receive buffer is not updated
  int errs = checkChunkValue<int>(recvBuf, recvTotalCount, -1);
  EXPECT_EQ(errs, 0) << "rank " << globalRank
                     << " checked receive buffer (expect no update) with "
                     << errs << " errors";

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

TEST_F(ctranAllToAllvTest, AllToAllvMultiBufs) {
  std::vector<int*> sendBufs(numRanks, nullptr), recvBufs(numRanks, nullptr);
  std::vector<void*> sendHdls(numRanks, nullptr), recvHdls(numRanks, nullptr);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctran);

  generateDistRandomCountsDisps();
  generateDistRandomExpValue();

  // Allocate different buffer for each send/recv chunk, and re-generate
  // displacement as offset to first buffer
  for (int i = 0; i < numRanks; ++i) {
    sendBufs[i] =
        (int*)createDataBuf(sendCounts[i] * sizeof(int), &sendHdls[i]);
    sendDisps[i] = (i == 0) ? 0 : sendBufs[i] - sendBufs[0];
    recvBufs[i] =
        (int*)createDataBuf(recvCounts[i] * sizeof(int), &recvHdls[i]);
    recvDisps[i] = (i == 0) ? 0 : recvBufs[i] - recvBufs[0];
  }

  sendBuf = sendBufs[0];
  recvBuf = recvBufs[0];
  run();

  for (int i = 0; i < numRanks; ++i) {
    releaseDataBuf(sendBufs[i], sendHdls[i]);
    releaseDataBuf(recvBufs[i], recvHdls[i]);
  }
}

TEST_F(ctranAllToAllvTest, AllToAllvDynamicRegister) {
  std::vector<int*> sendBufs(numRanks, nullptr), recvBufs(numRanks, nullptr);
  std::vector<void*> sendHdls(numRanks, nullptr), recvHdls(numRanks, nullptr);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctran);

  generateDistRandomCountsDisps();
  generateDistRandomExpValue();

  // Allocate different buffer for each send/recv chunk, and re-generate
  // displacement as offset to first buffer
  // Skip registration as for dynamic registration test
  for (int i = 0; i < numRanks; ++i) {
    sendBufs[i] = (int*)createDataBuf(sendCounts[i] * sizeof(int), nullptr);
    sendDisps[i] = (i == 0) ? 0 : sendBufs[i] - sendBufs[0];
    recvBufs[i] = (int*)createDataBuf(recvCounts[i] * sizeof(int), nullptr);
    recvDisps[i] = (i == 0) ? 0 : recvBufs[i] - recvBufs[0];
  }
  sendBuf = sendBufs[0];
  recvBuf = recvBufs[0];
  run();

  for (int i = 0; i < numRanks; ++i) {
    releaseDataBuf(sendBufs[i], nullptr);
    releaseDataBuf(recvBufs[i], nullptr);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
