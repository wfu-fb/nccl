// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef DDA_PRIVATE_MD_H_
#define DDA_PRIVATE_MD_H_

#include <unordered_map>
#include <vector>
#include "checks.h"
#include "collectives.h"
#include "ddaThreadSharedMd.h"

typedef enum {
  NCCL_DDA_TOPO_TYPE__NVS,
  NCCL_DDA_TOPO_TYPE__HCM,
  NCCL_DDA_TOPO_TYPE__UNKNOWN,
} ncclDDATopoType_t;

struct ncclComm;

class ddaPrivateMd {
public:
  ddaPrivateMd(ddaThreadSharedMd *threadSharedMd, ncclComm *comm);
  ~ddaPrivateMd();

  // flag indicating that each rank has arrived at the barrier
  uintptr_t barrierFlag;

  // barrier mailbox ID to use
  int barrierMboxId;

  ncclComm *comm;

  // device properties
  cudaDeviceProp devProp;

  // thread-shared meta-data
  ddaThreadSharedMd *threadSharedMd;

  // topology information
  ncclDDATopoType_t topoType;
  struct {
    struct {
      std::vector<int> gpus;
    } nvs;
    struct {
      struct {
        std::vector<int> gpus;
      } clique[2];
    } hcm;
  } u;

  struct commMd *commMdHost;
  struct commMd *commMdDev;

  std::unordered_map<int, int> rankToGpu;

private:
  // memory handles
  class ddaMemHandles *memHandles;
};

#endif
