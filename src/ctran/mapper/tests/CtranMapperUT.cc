
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <memory>
#include "CtranMapper.h"
#include "checks.h"
#include "comm.h"
#include "nccl_cvars.h"

class CtranMapperTest : public ::testing::Test {
 public:
  std::unique_ptr<CtranMapper> mapper;
  ncclComm* dummyComm;
  void* buf, *buf2;
  size_t bufSize = 4 * sizeof(char);
  void* hdl = nullptr;
  int cudaDev = 0;
  CtranMapperTest() = default;

 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_REGISTER", "none", 0);
    setenv("NCCL_CTRAN_BACKENDS", "", 0);
    ncclCvarInit();
    dummyComm = new ncclComm;
    dummyComm->rank = 0;
    dummyComm->nRanks = 1;
    dummyComm->commHash = 0;

    CUDACHECKABORT(cudaSetDevice(cudaDev));
    CUDACHECKABORT(cudaMalloc(&buf, bufSize));
    CUDACHECKABORT(cudaMalloc(&buf2, bufSize));
    CUDACHECKIGNORE(cudaMemset(buf, 0, bufSize));
  }
  void TearDown() override {
    delete dummyComm;
    CUDACHECKIGNORE(cudaFree(buf));
    CUDACHECKIGNORE(cudaFree(buf2));
    unsetenv("NCCL_CTRAN_REGISTER");
    unsetenv("NCCL_CTRAN_BACKENDS");
  }
};

TEST_F(CtranMapperTest, Init) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());
}

TEST_F(CtranMapperTest, regMemLazy) {
  setenv("NCCL_CTRAN_REGISTER", "lazy", 1);
  ncclCvarInit();

  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_THAT(hdl, testing::NotNull());
  unsetenv("NCCL_CTRAN_REGISTER");
}

TEST_F(CtranMapperTest, deregMem) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, ncclSuccess);
}

TEST_F(CtranMapperTest, deregMemNull) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  auto res = mapper->deregMem(hdl);
  EXPECT_EQ(res, ncclSuccess);
}

TEST_F(CtranMapperTest, doubleRegMem) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  void *hdl2;
  res = mapper->regMem(buf, bufSize, &hdl2, false);

  EXPECT_EQ(res, ncclSuccess);
  // The same handle should be returned
  EXPECT_EQ(hdl, hdl2);

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, ncclSuccess);
}

TEST_F(CtranMapperTest, doubleDeregMem) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, ncclSuccess);

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, ncclInvalidUsage);
}

TEST_F(CtranMapperTest, regMemEager) {
  setenv("NCCL_CTRAN_REGISTER", "eager", 1);
  ncclCvarInit();

  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, ncclSuccess);
  unsetenv("NCCL_CTRAN_REGISTER");
}

TEST_F(CtranMapperTest, regMemForce) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  auto res = mapper->regMem(buf, bufSize, &hdl, true);

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, ncclSuccess);
}

TEST_F(CtranMapperTest, regMemNMissingDereg) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_THAT(hdl, testing::NotNull());
}

TEST_F(CtranMapperTest, searchRegHandleMiss) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());
  bool dynamicRegist = false;
  auto res = mapper->searchRegHandle(buf, bufSize, &hdl, &dynamicRegist);

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  // upon cache miss, the buffer should be registered dynamically
  EXPECT_TRUE(dynamicRegist);

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, ncclSuccess);
}

TEST_F(CtranMapperTest, searchRegHandleHit) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  auto res = mapper->regMem(buf, bufSize, &hdl, false);

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  hdl = nullptr;
  bool dynamicRegist = false;
  res = mapper->searchRegHandle(buf, bufSize, &hdl, &dynamicRegist);

  EXPECT_EQ(res, ncclSuccess);
  EXPECT_THAT(hdl, testing::NotNull());

  EXPECT_FALSE(dynamicRegist);

  res = mapper->deregMem(hdl);
  EXPECT_EQ(res, ncclSuccess);
}

TEST_F(CtranMapperTest, icopy) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  CtranMapperRequest* req = new CtranMapperRequest(mapper.get());
  char* srcBuf;
  CUDACHECKIGNORE(cudaMalloc(&srcBuf, bufSize));
  CUDACHECKIGNORE(cudaMemset(srcBuf, 1, bufSize));

  auto res = mapper->icopy(buf, srcBuf, bufSize, &req);
  EXPECT_EQ(res, ncclSuccess);

  bool isComplete = false;
  do {
    req->test(&isComplete);
  } while (!isComplete);

  std::vector<char> observedVals(bufSize);
  CUDACHECKIGNORE(cudaMemcpy(observedVals.data(), buf, bufSize, cudaMemcpyDefault));
  EXPECT_THAT(observedVals, testing::ElementsAre(1, 1, 1, 1));

  CUDACHECKIGNORE(cudaFree(srcBuf));
}

TEST_F(CtranMapperTest, ReqInit) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  CtranMapperRequest* req = new CtranMapperRequest(mapper.get());
  EXPECT_THAT(req, testing::NotNull());

  delete req;
}

TEST_F(CtranMapperTest, ReqTest) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  CtranMapperRequest* req = new CtranMapperRequest(mapper.get());
  EXPECT_THAT(req, testing::NotNull());

  bool isComplete = false;
  auto res = req->test(&isComplete);
  EXPECT_EQ(res, ncclSuccess);
  EXPECT_TRUE(isComplete);

  delete req;
}

TEST_F(CtranMapperTest, ReqWait) {
  mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  CtranMapperRequest* req = new CtranMapperRequest(mapper.get());
  EXPECT_THAT(req, testing::NotNull());

  auto res = req->wait();
  EXPECT_EQ(res, ncclSuccess);

  delete req;
}
