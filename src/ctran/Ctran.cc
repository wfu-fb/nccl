#include "Ctran.h"
#include <nccl.h>
#include "CtranGpe.h"
#include "CtranMapper.h"
#include "argcheck.h"
#include "comm.h"
#include "nccl.h"
#include "nccl_cvars.h"

Ctran::Ctran(ncclComm* comm) {
  this->mapper = std::unique_ptr<CtranMapper>(new CtranMapper(comm));
  this->gpe = std::unique_ptr<CtranGpe>(new CtranGpe(comm->cudaDev));
}

ncclResult_t ctranInit(ncclComm* comm) {
  comm->ctran = std::unique_ptr<Ctran>(new Ctran(comm));
  return ncclSuccess;
}

bool ctranInitialized(ncclComm* comm) {
  return comm && comm->ctran && comm->ctran->mapper && comm->ctran->gpe;
}

ncclResult_t ctranDestroy(ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  // Return error if ctran is not fully initialized
  if (!ctranInitialized(comm)) {
    WARN("Ctran is not initialized\n");
    ret = ncclInternalError;
  }

  // Always cleanup resource that have been initialized
  comm->ctran.reset();

  return ret;
}
