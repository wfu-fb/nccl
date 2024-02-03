// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <memory>
#include "checks.h"
#include "comm.h"
#include "core.h"
#include "tests_common.cuh"
#include "cudawrapper.h"

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

#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
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

    cudaWrapper_ = ncclSetupWrappers(false);

    srand(time(NULL));
    expectedVal = rand();

    sendbuf = recvbuf = nullptr;
    sendBytes = count * sizeof(char);
    recvBytes = sendBytes * this->numRanks;

    CUDACHECKIGNORE(cudaWrapper->cudaSetDevice(this->localRank));
    CUDACHECKIGNORE(cudaWrapper->cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDACHECKIGNORE(cudaWrapper->cudaMalloc((void **)&sendbuf, sendBytes));
    CUDACHECKIGNORE(cudaWrapper->cudaMalloc((void **)&recvbuf, recvBytes));
    CUDACHECKIGNORE(
        cudaWrapper->cudaMemset(sendbuf, expectedVal * this->globalRank, sendBytes));
    CUDACHECKIGNORE(cudaWrapper->cudaMemset(recvbuf, rand(), recvBytes));
    // correct data for in-place allgather
    CUDACHECKIGNORE(cudaWrapper->cudaMemset(
        (char*)recvbuf + this->globalRank * sendBytes,
        expectedVal * this->globalRank,
        sendBytes));

    CUDACHECKIGNORE(cudaWrapper->cudaDeviceSynchronize());
  }
  void TearDown() override {}

  int localRank{0};
  int globalRank{0};
  int numRanks{0};

 private:
  CudaWrapper* cudaWrapper_;
};

TEST_F(FP8Test, ncclFp8E5M2SendRecv) {
  ncclDataType_t dt = ncclFp8E5M2;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res = ncclSuccess;
  constexpr int sendRank = 0;
  constexpr int recvRank = 1;
  if (this->globalRank == sendRank) {
    res = ncclSend(sendbuf, count, dt, recvRank, comm, stream);
  } else if (this->globalRank == recvRank) {
    res = ncclRecv(recvbuf, count, dt, sendRank, comm, stream);
  }
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaWrapper->cudaWrapper_->cudaStreamSynchronize(stream));

  if (this->globalRank == recvRank) {
    std::vector<char> observedVals(sendBytes, rand());
    CUDACHECKIGNORE(cudaWrapper->cudaMemcpy(
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

  CUDACHECKIGNORE(cudaWrapper->cudaWrapper_->cudaStreamSynchronize(stream));

  if (this->globalRank == recvRank) {
    std::vector<char> observedVals(sendBytes, rand());
    CUDACHECKIGNORE(cudaWrapper->cudaMemcpy(
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

  CUDACHECKIGNORE(cudaWrapper->cudaWrapper_->cudaStreamSynchronize(stream));

  for (int i = 0; i < this->numRanks; ++i) {
    std::vector<char> observedVals(sendBytes, rand());
    CUDACHECKIGNORE(cudaWrapper->cudaMemcpy(
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

  CUDACHECKIGNORE(cudaWrapper->cudaWrapper_->cudaStreamSynchronize(stream));

  for (int i = 0; i < this->numRanks; ++i) {
    std::vector<char> observedVals(sendBytes, rand());
    CUDACHECKIGNORE(cudaWrapper->cudaMemcpy(
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

  CUDACHECKIGNORE(cudaWrapper->cudaWrapper_->cudaStreamSynchronize(stream));

  std::vector<char> observedVals(sendBytes, rand());
  CUDACHECKIGNORE(cudaWrapper->cudaMemcpy(
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

  CUDACHECKIGNORE(cudaWrapper->cudaWrapper_->cudaStreamSynchronize(stream));

  std::vector<char> observedVals(sendBytes, rand());
  CUDACHECKIGNORE(cudaWrapper->cudaMemcpy(
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

  CUDACHECKIGNORE(cudaWrapper->cudaWrapper_->cudaStreamSynchronize(stream));

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

  CUDACHECKIGNORE(cudaWrapper->cudaWrapper_->cudaStreamSynchronize(stream));

  // FIXME: Don't check the result since there could be rounding errors

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(FP8Test, ncclFp8E5M2Reduce) {
  ncclDataType_t dt = ncclFp8E5M2;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);

  auto res =
      ncclReduce(sendbuf, recvbuf, count, dt, ncclSum, root, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaWrapper->cudaWrapper_->cudaStreamSynchronize(stream));

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

  CUDACHECKIGNORE(cudaWrapper->cudaWrapper_->cudaStreamSynchronize(stream));

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

  CUDACHECKIGNORE(cudaWrapper->cudaWrapper_->cudaStreamSynchronize(stream));

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

  CUDACHECKIGNORE(cudaWrapper->cudaWrapper_->cudaStreamSynchronize(stream));

  // FIXME: Don't check the result since there could be rounding errors

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
