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

class CtranTest : public ::testing::Test {
 public:
  CtranTest() = default;
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
    srand(time(NULL));
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

TEST_F(CtranTest, sendRecv) {
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctran);

  auto res = ncclSuccess;
  size_t count = 4096;
  ncclDataType_t dt = ncclFloat32;
  size_t bufSize = count * ncclTypeSize(dt);
  int sendRank = 0;
  int recvRank = 1;
  char* buf;
  cudaStream_t stream = 0;
  void* hdl;
  CUDACHECKIGNORE(cudaMalloc(&buf, bufSize));

  NCCLCHECK_TEST(ncclCommRegister(comm, buf, bufSize, &hdl));

  if (this->globalRank == sendRank) {
    CUDACHECKIGNORE(cudaMemset(buf, 1, bufSize));
    for (int i = 0; i < this->numRanks; ++i) {
      if (i != this->globalRank) {
        res = ctranSend(buf, count, dt, i, comm, stream);
        EXPECT_EQ(res, ncclSuccess);
      }
    }
  } else {
    CUDACHECKIGNORE(cudaMemset(buf, rand(), bufSize));
    res = ctranRecv(buf, count, dt, sendRank, comm, stream);
    EXPECT_EQ(res, ncclSuccess);
  }

  res = ctranGroupEndHook();
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  std::vector<char> observedVals(bufSize);
  CUDACHECKIGNORE(
      cudaMemcpy(observedVals.data(), buf, bufSize, cudaMemcpyDefault));
  EXPECT_THAT(observedVals, testing::Each(1));

  NCCLCHECK_TEST(ncclCommDeregister(comm, hdl));

  CUDACHECKIGNORE(cudaFree(buf));

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
