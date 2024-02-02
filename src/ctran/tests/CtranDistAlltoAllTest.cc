// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstdio>
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

class CtranAllToAllTest : public ::testing::Test {
 public:
  CtranAllToAllTest() = default;

  void generateDistRandomExpValue() {
    if (this->globalRank == 0) {
      expectedVal = rand();
    }
    MPI_Bcast(&expectedVal, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  void* createDataBuf(size_t nbytes, void** handle) {
    void* buf = nullptr;
    // Allocate data buffer, and assign different value for each send chunk
    CUDACHECK_TEST(cudaMalloc(&buf, nbytes));
    if (buf) {
      CUDACHECKIGNORE(cudaMemset(buf, -1, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());
      if (handle) {
        NCCLCHECK_TEST(ncclCommRegister(comm, buf, nbytes, handle));
      }
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
    CUDACHECKIGNORE(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyDefault));
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

  bool checkTestPrerequisite(void* sendBuf, void* recvBuf, size_t count) {
    EXPECT_NE(nullptr, comm);
    EXPECT_NE(nullptr, comm->ctran);
    if (!ctranAllToAllSupport(count, ncclInt, comm)) {
      if (this->globalRank == 0) {
        printf("Skip test because ctranAllToAllSupport returns false\n");
      }
      return false;
    }
    return true;
  }

  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
    comm = createNcclComm(this->globalRank, this->numRanks, this->localRank);

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
  }

  void
  run(const size_t count, const size_t bufCount, bool registerFlag = true) {
    int *sendBuf = nullptr, *recvBuf = nullptr;
    void *sendHdl = nullptr, *recvHdl = nullptr;

    assert(count * this->numRanks <= bufCount);
    generateDistRandomExpValue();

    // Allocate data buffer and register
    sendBuf = (int*)createDataBuf(
        bufCount * sizeof(int), registerFlag ? &sendHdl : nullptr);
    recvBuf = (int*)createDataBuf(
        bufCount * sizeof(int), registerFlag ? &recvHdl : nullptr);

    // Assign different value for each send chunk
    for (int i = 0; i < this->numRanks; ++i) {
      assignChunkValue<int>(
          sendBuf + i * count,
          count,
          this->expectedVal + this->globalRank * 100 + i + 1);
    }

    if (checkTestPrerequisite(sendBuf, recvBuf, count)) {
      // Run communication
      auto res = ctranAllToAll(sendBuf, recvBuf, count, ncclInt, comm, stream);
      ASSERT_EQ(res, ncclSuccess);
      CUDACHECK_TEST(cudaStreamSynchronize(stream));

      // Check each received chunk
      for (int i = 0; i < this->numRanks; ++i) {
        int errs = checkChunkValue<int>(
            recvBuf + i * count,
            count,
            this->expectedVal + i * 100 + this->globalRank + 1);
        EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << i
                           << " at " << recvBuf + i * count << " with " << errs
                           << " errors";
      }
      // Check remaining chunks in receive buffer is not updated
      if (count * this->numRanks < bufCount) {
        int errs = checkChunkValue<int>(
            recvBuf + count * this->numRanks,
            bufCount - count * this->numRanks,
            -1);
        EXPECT_EQ(errs, 0) << "rank " << globalRank
                           << " checked remaining chunk at "
                           << recvBuf + count * this->numRanks << " with "
                           << errs << " errors";
      }
    }

    releaseDataBuf(sendBuf, registerFlag ? sendHdl : nullptr);
    releaseDataBuf(recvBuf, registerFlag ? recvHdl : nullptr);
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};

  cudaStream_t stream{0};
  ncclComm_t comm{nullptr};
  int expectedVal{0};
};

TEST_F(CtranAllToAllTest, AllToAll) {
  run(8192, 8192 * this->comm->nRanks);
}

TEST_F(CtranAllToAllTest, UnalignedAllToAll) {
  run(9991, 9991 * this->comm->nRanks);
}

TEST_F(CtranAllToAllTest, SmallAllToAll) {
  // Even for small data transfer size, need buffer size >= pagesize for IB
  // registration
  run(2, 8192 * this->comm->nRanks);
}

TEST_F(CtranAllToAllTest, LargeAllToAll) {
  run(1024 * 1024 * 128UL, 1024 * 1024 * 128UL * this->comm->nRanks);
}

TEST_F(CtranAllToAllTest, ZeroByteAllToAll) {
  run(0, 8192 * this->comm->nRanks);
}

TEST_F(CtranAllToAllTest, AllToAllDynamicRegister) {
  run(8192, 8192 * this->comm->nRanks, false);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
