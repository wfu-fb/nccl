// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef DDA_MEM_HANDLES_H_
#define DDA_MEM_HANDLES_H_

#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "checks.h"
#include "ddaThreadSharedMd.h"

struct ncclComm;

constexpr int MEM_HANDLE_NAME_LEN_MAX = 256;

struct IpcHandleState {
  std::string name;
  void *addr;
  cudaIpcMemHandle_t handle;
  bool isMmapped;
};

class ddaMemHandles {
public:
  ddaMemHandles(ddaThreadSharedMd *threadSharedMd, ncclComm *comm);
  ~ddaMemHandles();
  ncclResult_t insertMemHandle(void *addr, std::string name);
  ncclResult_t exchangeMemHandles(void);
  void *getMemAddr(int rank, std::string name);

private:
  ddaThreadSharedMd *threadSharedMd;
  ncclComm *comm;
  std::unordered_map<int, std::vector<IpcHandleState>> allMemHandles;
};

#endif
