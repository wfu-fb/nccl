// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include "Ctran.h"
#include "CtranAlgoDev.h"
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

TEST_F(CtranTest, AlgoDeviceState) {
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctran);

  ASSERT_NE(nullptr, comm->ctran->algo->devState_d);

  // check contents of devState_d to make sure it is initialized correctly
  CtranAlgoDeviceState devState;
  CUDACHECK_TEST(cudaMemcpy(
      &devState,
      comm->ctran->algo->devState_d,
      sizeof(devState),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(devState.localRanks, comm->localRanks);
  EXPECT_EQ(devState.localRank, comm->localRank);
  EXPECT_EQ(devState.bufSize, NCCL_CTRAN_SHARED_DEVBUF_SIZE);

  for (int i = 0; i < comm->localRanks; i++) {
    EXPECT_EQ(devState.localRankToRank[i], comm->localRankToRank[i]);
    for (int j = 0; j < comm->localRanks; j++) {
      if (i == j) {
        // Expect null for owner itself
        EXPECT_EQ(devState.allPeerToBufsMap[i][j], nullptr);
        EXPECT_EQ(devState.allPeerToBufStatesMap[i][j], nullptr);
      } else {
        // Expect IPC buffer is allocated and state is reset for all peers
        EXPECT_NE(devState.allPeerToBufsMap[i][j], nullptr);
        EXPECT_NE(devState.allPeerToBufStatesMap[i][j], nullptr);

        // Copy buffer state to host and check values are reset to default
        struct CtranAlgoDeviceBufState stateVal;
        CUDACHECK_TEST(cudaMemcpy(
            &stateVal,
            devState.allPeerToBufStatesMap[i][j],
            sizeof(stateVal),
            cudaMemcpyDeviceToHost));
        for (int k = 0; k < CTRAN_ALGO_MAX_THREAD_BLOCKS; k++) {
          EXPECT_EQ(stateVal.stepOnSameBlockIdx[k], CTRAN_ALGO_STEP_RESET);
        }
      }
    }
  }

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CtranTest, CommAbort) {
  ncclResult_t res;
  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctran);

  // Expect shared resource has been released properly
  res = ncclCommAbort(comm);
  ASSERT_EQ(res, ncclSuccess);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
