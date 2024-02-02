#include <gtest/gtest.h>
#include <nccl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdio>
#include <filesystem>
#include "nccl_cvars.h"
#include "CtranTopoFile.h"

namespace fs = std::filesystem;

class NCCLEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    // Turn off NCCL debug logging, allow user to turn on via command line
    setenv("NCCL_DEBUG", "WARN", 0);
  }
  ~NCCLEnvironment() override {}
};

class CvarTest : public ::testing::Test {
 public:
  CvarTest() = default;
};

TEST_F(CvarTest, TEST_READ_TOPO_FILE) {
  char filename[] = "/tmp/mytemp.XXXXXX";
  int fd = mkstemp(filename);

  setenv("NCCL_CTRAN_TOPO_FILE", filename, 1);
  setenv("NCCL_CTRAN_TOPO_FILE_KEYS", "KEYA,KEYB,KEYC", 1);

  // Deliberately putting these out of order to make sure we're tolerant to that
  // Also deliberately including KEYAB to ensure we don't just substring match
  char buf[64] = "KEYC=3\nKEYA=55\nKEYB=12\nKEYAB=212\n";
  write(fd, buf, 64);

  ncclCvarInit();
  auto vals = getTopoValsFromFile();
  EXPECT_EQ(vals.size(), 3);
  EXPECT_EQ(vals.at(0), 55);
  EXPECT_EQ(vals.at(1), 12);
  EXPECT_EQ(vals.at(2), 3);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new NCCLEnvironment);
  return RUN_ALL_TESTS();
}
