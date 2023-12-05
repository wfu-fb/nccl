// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include "Ctran.h"
#include "checks.h"
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

class CommRegisterTest : public ::testing::Test {
 public:
  CommRegisterTest() = default;
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

#ifdef NCCL_REGISTRATION_SUPPORTED
TEST_F(CommRegisterTest, Register) {
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  const int bufSize = 1048576;

  void* buf = nullptr;
  void* handle = nullptr;

  CUDACHECK_TEST(cudaMalloc(&buf, bufSize));

  EXPECT_EQ(ncclCommRegister(comm, buf, bufSize, &handle), ncclSuccess);
  EXPECT_NE(handle, nullptr);

  EXPECT_EQ(ncclCommDeregister(comm, handle), ncclSuccess);

  CUDACHECK_TEST(cudaFree(buf));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CommRegisterTest, CtranRegisterNone) {
  setenv("NCCL_CTRAN_REGISTER", "none", 1);
  ncclCvarInit();

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  const int bufSize = 1048576;

  void* buf = nullptr;
  void* handle = nullptr;

  CUDACHECK_TEST(cudaMalloc(&buf, bufSize));

  EXPECT_EQ(ncclCommRegister(comm, buf, bufSize, &handle), ncclSuccess);
  EXPECT_EQ(handle, nullptr);

  EXPECT_EQ(ncclCommDeregister(comm, handle), ncclSuccess);

  CUDACHECK_TEST(cudaFree(buf));
  NCCLCHECK_TEST(ncclCommDestroy(comm));
  unsetenv("NCCL_CTRAN_REGISTER");
}

TEST_F(CommRegisterTest, NoMapperCtranRegister) {
  // ensure NCCL_CTRAN_REGISTER has been reset to default
  ncclCvarInit();

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  const int bufSize = 1048576;

  // Mimic failed mapper initialization
  comm->ctran->mapper.reset();

  void* buf = nullptr;
  void* handle = nullptr;

  CUDACHECK_TEST(cudaMalloc(&buf, bufSize));

  EXPECT_EQ(ncclCommRegister(comm, buf, bufSize, &handle), ncclInvalidUsage);
  EXPECT_EQ(handle, nullptr);

  EXPECT_EQ(ncclCommDeregister(comm, handle), ncclInvalidUsage);

  CUDACHECK_TEST(cudaFree(buf));
  // ctran internal mapper has been reset, so commDestroy should fail
  ASSERT_EQ(ncclCommDestroy(comm), ncclInternalError);
}
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
