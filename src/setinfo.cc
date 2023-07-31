#include "nccl.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "utils.h"
#include <string.h>

NCCL_API(ncclResult_t, ncclCommSetInfo, ncclComm_t* comm, ncclConfig_t* config);
ncclResult_t ncclCommSetInfo(ncclComm_t* comm, ncclConfig_t* config) {
  ncclResult_t ret = ncclSuccess;
  int proto, algo;

  // Check protocol string from user
  if (config->protoStr != NCCL_CONFIG_UNDEF_PTR) {
    NCCLCHECK(strToEnum(config->protoStr, ncclProtoStr, &proto));
  }
  // No protocol string, reset to default
  else {
    proto = NCCL_CONFIG_UNDEF_INT;
  }

  // Handle algorithm string from user
  if (config->algoStr != NCCL_CONFIG_UNDEF_PTR) {
    NCCLCHECK(strToEnum(config->algoStr, ncclAlgoStr, &algo));
  }
  // No algorithm string, reset to default
  else {
    algo = NCCL_CONFIG_UNDEF_INT;
  }

  // Update proto/algo
  (*comm)->config.proto = proto;
  (*comm)->config.algo = algo;

  return ret;
}

NCCL_API(ncclResult_t, ncclCommGetInfo, ncclComm_t* comm, ncclConfig_t *config);
ncclResult_t ncclCommGetInfo(ncclComm_t *comm, ncclConfig_t *config) {
  ncclResult_t ret = ncclSuccess;
  config->proto = (*comm)->config.proto;
  if((*comm)->config.proto != NCCL_CONFIG_UNDEF_INT){
    config->protoStr = ncclProtoStr[(*comm)->config.proto];
  }
  config->algo = (*comm)->config.algo;
  if((*comm)->config.algo != NCCL_CONFIG_UNDEF_INT){
    config->algoStr = ncclAlgoStr[(*comm)->config.algo];
  }
  return ret;
}
