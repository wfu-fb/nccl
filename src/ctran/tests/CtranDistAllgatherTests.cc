// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
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

class CtranAllgatherTest : public ::testing::Test {
 public:
  CtranAllgatherTest() = default;
  char expectedVal;
  size_t count = 8192;
  ncclDataType_t dt = ncclBfloat16;
  size_t sendBytes, recvBytes;
  void *sendbuf, *recvbuf;
  void *sendHdl, *recvHdl;
  cudaStream_t stream = 0;

  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
    srand(time(NULL));
    expectedVal = rand();

    sendbuf = recvbuf = nullptr;
    sendHdl = recvHdl = nullptr;
    sendBytes = count * ncclTypeSize(dt);
    recvBytes = sendBytes * this->numRanks;

    CUDACHECKIGNORE(cudaSetDevice(this->localRank));
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

  void TearDown() override {
    CUDACHECKIGNORE(cudaFree(sendbuf));
    CUDACHECKIGNORE(cudaFree(recvbuf));
    CUDACHECKIGNORE(cudaStreamDestroy(stream));
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

TEST_F(CtranAllgatherTest, InplaceAllgatherDirect) {
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctran);

  NCCLCHECK_TEST(ncclCommRegister(comm, recvbuf, recvBytes, &recvHdl));

  void* inplaceSendBuf = (char*)recvbuf + this->globalRank * sendBytes;
  auto res =
      ctranAllGatherDirect(inplaceSendBuf, recvbuf, count, dt, comm, stream);
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

  NCCLCHECK_TEST(ncclCommDeregister(comm, recvHdl));

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CtranAllgatherTest, OutOfPlaceAllgatherDirect) {
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctran);

  NCCLCHECK_TEST(ncclCommRegister(comm, sendbuf, sendBytes, &sendHdl));
  NCCLCHECK_TEST(ncclCommRegister(comm, recvbuf, recvBytes, &recvHdl));

  auto res = ctranAllGatherDirect(sendbuf, recvbuf, count, dt, comm, stream);
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

  NCCLCHECK_TEST(ncclCommDeregister(comm, sendHdl));
  NCCLCHECK_TEST(ncclCommDeregister(comm, recvHdl));

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
