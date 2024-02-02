// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <climits>
#include <cstddef>
#include <iostream>
#include "CtranIb.h"
#include "comm.h"
#include "nccl_cvars.h"
#include "socket.h"
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

class CtranIbHcaTest : public ::testing::Test {
 public:
  CtranIbHcaTest() = default;
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
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
};

TEST_F(CtranIbHcaTest, IbHcaExactMatchDev) {
  this->printTestDesc(
      "IbHcaExactMatchDev",
      "Expect only exactly matched device lists specified by NCCL_IB_HCA are used.");

  int nDevices;
  CUDACHECK_TEST(cudaGetDeviceCount(&nDevices));

  std::string ibHcaStr =
      "=mlx5_10:1,mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_11:1";
  std::vector<std::string> ibHcaExactDevs{
      "mlx5_10",
      "mlx5_0",
      "mlx5_3",
      "mlx5_4",
      "mlx5_5",
      "mlx5_6",
      "mlx5_9",
      "mlx5_11"};
  setenv("NCCL_IB_HCA", ibHcaStr.c_str(), 1);

  // Rank 0 creates comm with differen local GPU, to check whether all used
  // devices match the condition
  nDevices = std::min(nDevices, (int)ibHcaExactDevs.size());
  for (int devId = 0; devId < nDevices; devId++) {
    int myDevId = this->globalRank == 0 ? devId : this->localRank;
    ncclComm_t comm = createNcclComm(this->globalRank, this->numRanks, myDevId);

    EXPECT_EQ(NCCL_IB_HCA_PREFIX, "=");

    try {
      auto ctranIb = std::unique_ptr<class CtranIb>(new class CtranIb(comm));
      EXPECT_EQ(ctranIb->getIbDevName(), ibHcaExactDevs[devId]);
      printf(
          "CtranIbTest.IbHcaExactMatchDev: Rank %d devId %d uses devName %s devPort %d\n",
          this->globalRank,
          devId,
          ctranIb->getIbDevName().c_str(),
          ctranIb->getIbDevPort());
    } catch (const std::bad_alloc& e) {
      printf("CtranIbTest: IB backend not enabled. Skip test\n");
    }
    NCCLCHECK_TEST(ncclCommDestroy(comm));
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
