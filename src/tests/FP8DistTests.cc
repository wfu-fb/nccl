// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <AlgoInit.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <cstdlib>
#include <memory>
#include "checks.h"
#include "comm.h"
#include "core.h"
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

#if defined(NCCL_ENABLE_FP8)
class FP8Test : public ::testing::Test {
 public:
  FP8Test() = default;
  char expectedVal;
  size_t count = 8192;
  size_t sendBytes, recvBytes;
  void *sendbuf, *recvbuf;
  cudaStream_t stream = 0;
  int root = 0;

  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    srand(time(NULL));
    expectedVal = rand();

    sendbuf = recvbuf = nullptr;
    sendBytes = count * sizeof(char);
    recvBytes = sendBytes * this->numRanks;

    CUDACHECKABORT(cudaSetDevice(this->localRank));
    CUDACHECKIGNORE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDACHECKIGNORE(cudaMalloc(&sendbuf, sendBytes));
    CUDACHECKIGNORE(cudaMalloc(&recvbuf, recvBytes));
    CUDACHECKIGNORE(
        cudaMemset(sendbuf, expectedVal * this->globalRank, sendBytes));
    CUDACHECKIGNORE(cudaMemset(recvbuf, rand(), recvBytes));
    // correct data for in-place allgather
    CUDACHECKIGNORE(cudaMemset(
        (char*)recvbuf + this->globalRank * sendBytes,
        expectedVal * this->globalRank,
        sendBytes));

    CUDACHECKIGNORE(cudaDeviceSynchronize());
  }
  void TearDown() override {}

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

TEST_F(FP8Test, ncclFp8E5M2SendRecv) {
  ncclDataType_t dt = ncclFp8E5M2;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res = ncclSuccess;
  int sendRank = 0;
  int recvRank = 1;
  if (this->globalRank == sendRank) {
    res = ncclSend(sendbuf, count, dt, recvRank, comm, stream);
  } else if (this->globalRank == recvRank) {
    res = ncclRecv(recvbuf, count, dt, sendRank, comm, stream);
  }
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  if (this->globalRank == recvRank) {
    std::vector<char> observedVals(sendBytes, rand());
    CUDACHECKIGNORE(cudaMemcpy(
        observedVals.data(), (char*)recvbuf, sendBytes, cudaMemcpyDefault));
    EXPECT_THAT(observedVals, testing::Each(expectedVal * sendRank));
  }

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(FP8Test, ncclFp8E4M3SendRecv) {
  ncclDataType_t dt = ncclFp8E4M3;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res = ncclSuccess;
  int sendRank = 0;
  int recvRank = 1;
  if (this->globalRank == sendRank) {
    res = ncclSend(sendbuf, count, dt, recvRank, comm, stream);
  } else if (this->globalRank == recvRank) {
    res = ncclRecv(recvbuf, count, dt, sendRank, comm, stream);
  }
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  if (this->globalRank == recvRank) {
    std::vector<char> observedVals(sendBytes, rand());
    CUDACHECKIGNORE(cudaMemcpy(
        observedVals.data(), (char*)recvbuf, sendBytes, cudaMemcpyDefault));
    EXPECT_THAT(observedVals, testing::Each(expectedVal * sendRank));
  }

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(FP8Test, ncclFp8E5M2Allgather) {
  ncclDataType_t dt = ncclFp8E5M2;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res = ncclAllGather(sendbuf, recvbuf, count, dt, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  for (int i = 0; i < this->numRanks; ++i) {
    std::vector<char> observedVals(sendBytes, rand());
    CUDACHECKIGNORE(cudaMemcpy(
        observedVals.data(),
        (char*)recvbuf + sendBytes * i,
        sendBytes,
        cudaMemcpyDefault));
    EXPECT_THAT(observedVals, testing::Each(expectedVal * i));
  }

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(FP8Test, ncclFp8E4M3AllGather) {
  ncclDataType_t dt = ncclFp8E4M3;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res = ncclAllGather(sendbuf, recvbuf, count, dt, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  for (int i = 0; i < this->numRanks; ++i) {
    std::vector<char> observedVals(sendBytes, rand());
    CUDACHECKIGNORE(cudaMemcpy(
        observedVals.data(),
        (char*)recvbuf + sendBytes * i,
        sendBytes,
        cudaMemcpyDefault));
    EXPECT_THAT(observedVals, testing::Each(expectedVal * i));
  }

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(FP8Test, ncclFp8E5M2Bcast) {
  ncclDataType_t dt = ncclFp8E5M2;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res = ncclBroadcast(sendbuf, recvbuf, count, dt, root, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  std::vector<char> observedVals(sendBytes, rand());
  CUDACHECKIGNORE(cudaMemcpy(
      observedVals.data(), (char*)recvbuf, sendBytes, cudaMemcpyDefault));
  EXPECT_THAT(observedVals, testing::Each(expectedVal * root));

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(FP8Test, ncclFp8E4M3AllBcast) {
  ncclDataType_t dt = ncclFp8E4M3;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res = ncclBroadcast(sendbuf, recvbuf, count, dt, root, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  std::vector<char> observedVals(sendBytes, rand());
  CUDACHECKIGNORE(cudaMemcpy(
      observedVals.data(), (char*)recvbuf, sendBytes, cudaMemcpyDefault));
  EXPECT_THAT(observedVals, testing::Each(expectedVal * root));

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(FP8Test, ncclFp8E5M2AllReduce) {
  ncclDataType_t dt = ncclFp8E5M2;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res = ncclAllReduce(sendbuf, recvbuf, count, dt, ncclSum, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  // FIXME: Don't check the result since there could be rounding errors

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(FP8Test, ncclFp8E4M3AllReduce) {
  ncclDataType_t dt = ncclFp8E4M3;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res = ncclAllReduce(sendbuf, recvbuf, count, dt, ncclSum, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  // FIXME: Don't check the result since there could be rounding errors

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(FP8Test, ncclFp8E5M2AllReduceDDA) {
  setenv("NCCL_ALLREDUCE_ALGO2", "dda", 1);
  ncclCvarInit();
  ncclDataType_t dt = ncclFp8E5M2;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res = ncclAllReduce(sendbuf, recvbuf, count, dt, ncclSum, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  // FIXME: Don't check the result since there could be rounding errors

  NCCLCHECK_TEST(ncclCommDestroy(comm));
  unsetenv("NCCL_ALLREDUCE_ALGO2");
  ncclCvarInit();
}

TEST_F(FP8Test, ncclFp8E4M3AllReduceDDA) {
  setenv("NCCL_ALLREDUCE_ALGO2", "dda", 1);
  ncclCvarInit();
  ncclDataType_t dt = ncclFp8E4M3;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res = ncclAllReduce(sendbuf, recvbuf, count, dt, ncclSum, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  // FIXME: Don't check the result since there could be rounding errors

  NCCLCHECK_TEST(ncclCommDestroy(comm));
  unsetenv("NCCL_ALLREDUCE_ALGO2");
  ncclCvarInit();
}

TEST_F(FP8Test, ncclFp8E5M2Reduce) {
  ncclDataType_t dt = ncclFp8E5M2;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res =
      ncclReduce(sendbuf, recvbuf, count, dt, ncclSum, root, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  // FIXME: Don't check the result since there could be rounding errors

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(FP8Test, ncclFp8E4M3Reduce) {
  ncclDataType_t dt = ncclFp8E4M3;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res =
      ncclReduce(sendbuf, recvbuf, count, dt, ncclSum, root, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  // FIXME: Don't check the result since there could be rounding errors

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(FP8Test, ncclFp8E5M2ReduceScatter) {
  ncclDataType_t dt = ncclFp8E5M2;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res =
      ncclReduceScatter(sendbuf, recvbuf, count, dt, ncclSum, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  // FIXME: Don't check the result since there could be rounding errors

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(FP8Test, ncclFp8E4M3ReduceScatter) {
  ncclDataType_t dt = ncclFp8E4M3;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res =
      ncclReduceScatter(sendbuf, recvbuf, count, dt, ncclSum, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  // FIXME: Don't check the result since there could be rounding errors

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
