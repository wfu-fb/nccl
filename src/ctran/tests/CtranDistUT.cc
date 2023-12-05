// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
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
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

TEST_F(CtranTest, Initialized) {
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctran);

  EXPECT_NE(nullptr, comm->ctran->mapper);
  EXPECT_NE(nullptr, comm->ctran->gpe);
  EXPECT_TRUE(ctranInitialized(comm));

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CtranTest, MapperNotInitialized) {
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctran);

  comm->ctran->mapper.reset();
  ASSERT_FALSE(ctranInitialized(comm));

  // Cleanup comm resource, but expect ncclInternalError because gpe is not
  // initialized
  ASSERT_EQ(ncclCommDestroy(comm), ncclInternalError);
}

TEST_F(CtranTest, GpeNotInitialized) {
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctran);

  comm->ctran->gpe.reset();
  ASSERT_FALSE(ctranInitialized(comm));

  // Cleanup comm resource, but expect ncclInternalError because gpe is not
  // initialized
  ASSERT_EQ(ncclCommDestroy(comm), ncclInternalError);
}

TEST_F(CtranTest, PostCommDestory) {
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctran);

  NCCLCHECK_TEST(ncclCommDestroy(comm));

  ASSERT_FALSE(ctranInitialized(comm));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
