// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include "../include/checks.h"

// Helper class to prepare required arguments for ncclAllreduceSparseBlock
class AllreduceSparseBlockArgs {
 public:
  AllreduceSparseBlockArgs(
      size_t inBlockCount,
      size_t inBlockLength,
      int64_t* inRecvIndices,
      size_t inRecvCount)
      : blockCount(inBlockCount),
        blockLength(inBlockLength),
        recvCount(inRecvCount) {
    CUDACHECKIGNORE(cudaSetDevice(0));
    CUDACHECKIGNORE(cudaStreamCreate(&stream));
    NCCLCHECKIGNORE(ncclCommInitAll(&comm, 1, {0}));
    CUDACHECKIGNORE(cudaMalloc(
        (void**)&sendBuff, sizeof(int32_t) * blockCount * blockLength));
    CUDACHECKIGNORE(cudaMalloc((void**)&recvBuff, sizeof(int32_t) * recvCount));
    CUDACHECKIGNORE(
        cudaMalloc((void**)&recvIndices, sizeof(int64_t) * blockCount));

    CUDACHECKIGNORE(cudaMemcpy(
        recvIndices,
        inRecvIndices,
        sizeof(int64_t) * blockCount,
        cudaMemcpyHostToDevice));
  }

  ~AllreduceSparseBlockArgs() {
    CUDACHECKIGNORE(cudaSetDevice(0));
    CUDACHECKIGNORE(cudaFree(sendBuff));
    CUDACHECKIGNORE(cudaFree(recvBuff));
    CUDACHECKIGNORE(cudaFree(recvIndices));
    NCCLCHECKIGNORE(ncclCommDestroy(comm));
    CUDACHECKIGNORE(cudaStreamDestroy(stream));
  }

  // user passed arguments
  const size_t blockCount{0};
  const size_t blockLength{0};
  const size_t recvCount{0};

  // created arguments
  int32_t* sendBuff{nullptr};
  int32_t* recvBuff{nullptr};
  int64_t* recvIndices{nullptr};
  cudaStream_t stream;
  ncclComm_t comm;
};

class AllreduceSparseBlockArgCheckTest : public ::testing::Test {
 public:
  AllreduceSparseBlockArgCheckTest() {
    // Turn on pointer check before executing any argument check test.
    setenv("NCCL_CHECK_POINTERS", "1", 1);
  }
};

TEST_F(AllreduceSparseBlockArgCheckTest, UnsupportedOp) {
  std::vector<int64_t> recvIndices{0, 8, 84, 128};
  auto args = std::make_unique<AllreduceSparseBlockArgs>(
      recvIndices.size(), 4, recvIndices.data(), 256);
  EXPECT_EQ(
      ncclAllReduceSparseBlock(
          args->sendBuff,
          args->recvIndices,
          args->blockCount,
          args->blockLength,
          args->recvBuff,
          args->recvCount,
          ncclInt,
          ncclMax,
          args->comm,
          args->stream),
      ncclInvalidArgument);
}

TEST_F(AllreduceSparseBlockArgCheckTest, InPlaceBuff) {
  std::vector<int64_t> recvIndices{0, 8, 84, 128};
  auto args = std::make_unique<AllreduceSparseBlockArgs>(
      recvIndices.size(), 4, recvIndices.data(), 256);
  EXPECT_EQ(
      ncclAllReduceSparseBlock(
          args->recvBuff, /* same buffer for both send and recv */
          args->recvIndices,
          args->blockCount,
          args->blockLength,
          args->recvBuff,
          args->recvCount,
          ncclInt,
          ncclSum,
          args->comm,
          args->stream),
      ncclInvalidArgument);
}

TEST_F(AllreduceSparseBlockArgCheckTest, InvalidSize) {
  std::vector<int64_t> recvIndices{0, 8, 84, 128};
  auto args = std::make_unique<AllreduceSparseBlockArgs>(
      recvIndices.size(),
      8,
      recvIndices.data(),
      16 /*recv_count < block_count * block_length*/);
  EXPECT_EQ(
      ncclAllReduceSparseBlock(
          args->sendBuff,
          args->recvIndices,
          args->blockCount,
          args->blockLength,
          args->recvBuff,
          args->recvCount,
          ncclInt,
          ncclSum,
          args->comm,
          args->stream),
      ncclInvalidArgument);
}

TEST_F(AllreduceSparseBlockArgCheckTest, InvalidSendBuff) {
  int invalidDevBuff[4];
  std::vector<int64_t> recvIndices{0, 8, 84, 128};
  auto args = std::make_unique<AllreduceSparseBlockArgs>(
      recvIndices.size(), 4, recvIndices.data(), 256);

  EXPECT_EQ(
      ncclAllReduceSparseBlock(
          (void*)invalidDevBuff,
          args->recvIndices,
          args->blockCount,
          args->blockLength,
          args->recvBuff,
          args->recvCount,
          ncclInt,
          ncclSum,
          args->comm,
          args->stream),
      ncclInvalidArgument);
}

TEST_F(AllreduceSparseBlockArgCheckTest, InvalidRecvBuff) {
  int invalidDevBuff[4];
  std::vector<int64_t> recvIndices{0, 8, 84, 128};
  auto args = std::make_unique<AllreduceSparseBlockArgs>(
      recvIndices.size(), 4, recvIndices.data(), 256);

  EXPECT_EQ(
      ncclAllReduceSparseBlock(
          args->sendBuff,
          args->recvIndices,
          args->blockCount,
          args->blockLength,
          (void*)invalidDevBuff,
          args->recvCount,
          ncclInt,
          ncclSum,
          args->comm,
          args->stream),
      ncclInvalidArgument);
}

TEST_F(AllreduceSparseBlockArgCheckTest, InvalidRecvIndices) {
  std::vector<int64_t> recvIndices{0, 8, 84, 128};
  auto args = std::make_unique<AllreduceSparseBlockArgs>(
      recvIndices.size(), 4, recvIndices.data(), 256);

  EXPECT_EQ(
      ncclAllReduceSparseBlock(
          args->sendBuff,
          recvIndices.data() /* Invalid recvIndices device pointer */,
          args->blockCount,
          args->blockLength,
          args->recvBuff,
          args->recvCount,
          ncclInt,
          ncclSum,
          args->comm,
          args->stream),
      ncclInvalidArgument);
}
