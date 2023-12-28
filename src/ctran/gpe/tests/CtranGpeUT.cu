
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <iostream>
#include "CtranGpe.h"
#include "CtranGpeImpl.h"
#include "CtranGpeKernel.h"
#include "checks.h"

class CtranGpeTest : public ::testing::Test {
 public:
  CtranGpe* gpe;
  int cudaDev;
  ncclComm_t dummyComm;
  CtranGpeTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    gpe = nullptr;
  }
  void TearDown() override {
    if (gpe != nullptr) {
      delete gpe;
    }
  }
};

class CtranGpeKernelTest : public ::testing::Test {
 public:
  volatile int* testFlag;
  int cudaDev;
  CtranGpeKernelTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    CUDACHECKIGNORE(cudaSetDevice(cudaDev));
    CUDACHECKIGNORE(cudaHostAlloc((void**) &testFlag, sizeof(int), cudaHostAllocDefault));
    *testFlag = UNSET;
  }
  void TearDown() override {
    CUDACHECKIGNORE(cudaFreeHost((void*)testFlag));
  }
};

constexpr std::string_view kExpectedOutput = "CtranGpeTestAlgoFunc Called";
static ncclResult_t CtranGpeTestAlgoFunc(
    std::vector<std::unique_ptr<struct OpElem>> opGroup) {
  std::cout << kExpectedOutput;
  return ncclSuccess;
}

__global__ void CtranGpeTestKernel(int *flag) {
  ncclKernelStallStream(flag);
}

TEST_F(CtranGpeTest, gpeThread) {
  gpe = new CtranGpe(cudaDev);
  EXPECT_THAT(gpe, testing::NotNull());
}

TEST_F(CtranGpeTest, SubmitOpBadArgs) {
  ncclResult_t res = ncclSuccess;
  CtranGpe* gpe = new CtranGpe(cudaDev);

  std::vector<std::unique_ptr<struct OpElem>> ops;
  struct OpElem* op;
  op = new struct OpElem(OpElem::opType::SEND, nullptr, dummyComm);
  op->send.sendbuff = nullptr;
  op->send.count = 0;
  op->send.datatype = ncclInt8;
  op->send.peerRank = 0;
  ops.push_back(std::unique_ptr<struct OpElem>(op));

  /* NOTE: invalid CUDA kernel should return error code */
  res = gpe->submit(std::move(ops), &CtranGpeTestAlgoFunc, nullptr);

  EXPECT_NE(res, ncclSuccess);
}

TEST_F(CtranGpeTest, SubmitOp) {
  ncclResult_t res = ncclSuccess;
  CtranGpe* gpe = new CtranGpe(cudaDev);

  std::vector<std::unique_ptr<struct OpElem>> ops;
  struct OpElem* op;
  op = new struct OpElem(OpElem::opType::RECV, nullptr, dummyComm);
  op->recv.recvbuff = nullptr;
  op->recv.count = 0;
  op->recv.datatype = ncclInt8;
  op->recv.peerRank = 0;
  ops.push_back(std::unique_ptr<struct OpElem>(op));

  testing::internal::CaptureStdout();
  res = gpe->submit(
      std::move(ops),
      &CtranGpeTestAlgoFunc,
      reinterpret_cast<void*>(CtranGpeTestKernel));

  EXPECT_EQ(res, ncclSuccess);

  ops.clear();
  delete op;
  delete gpe;
  gpe = nullptr;

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput));
}

TEST_F(CtranGpeKernelTest, launchTerminateStallKernel) {
    dim3 grid = { 1, 1, 1 };
    dim3 blocks = { 1, 1, 1 };
    void *args[] = { &testFlag };
    auto res = cudaLaunchKernel(reinterpret_cast<void*>(CtranGpeTestKernel), grid, blocks, args, 0, 0);

    EXPECT_EQ(res, cudaSuccess);

    while (*testFlag != KERNEL_STARTED) {
        EXPECT_THAT(*testFlag, testing::Not(KERNEL_TERMINATE));
    }

    *testFlag = KERNEL_TERMINATE;
    res = cudaStreamSynchronize(0);

    EXPECT_EQ(res, cudaSuccess);
}
