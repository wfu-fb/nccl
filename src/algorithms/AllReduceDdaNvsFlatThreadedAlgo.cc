// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "AllReduceDdaNvsFlatThreadedAlgo.h"

#include "AlgoUtils.h"
#include "comm.h"
#include "debug.h"

namespace nccl {
namespace algorithms {

AllReduceDdaNvsFlatThreadedAlgo::AllReduceDdaNvsFlatThreadedAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream,
    const DdaDeviceState* devStates_d,
    uintptr_t barrierFlag)
    : sendbuff_(sendbuff),
      recvbuff_(recvbuff),
      count_(count),
      datatype_(datatype),
      op_(op),
      comm_(comm),
      stream_(stream),
      devStates_d_(devStates_d),
      barrierFlag_(barrierFlag) {}

AllReduceDdaNvsFlatThreadedAlgo::~AllReduceDdaNvsFlatThreadedAlgo() {}

template <typename T>
ncclResult_t AllReduceDdaNvsFlatThreadedAlgo::launchKernel() {
  const void* func = nullptr;
  ASSIGN_FUNC_NRANKS(func, ncclKernel_AllReduce_DDA2_Flat, comm_->nRanks);

  dim3 grid;
  dim3 blocks;
  grid.x = 1;
  grid.y = 1;
  grid.z = 1;

  blocks.x = 128;
  blocks.y = 1;
  blocks.z = 1;

  void* args[] = {
      &barrierFlag_,
      &devStates_d_,
      &comm_->rank,
      &sendbuff_,
      &recvbuff_,
      &count_};
  CUDACHECK(cudaLaunchKernel(func, grid, blocks, args, 0, stream_));
  return ncclSuccess;
}

ncclResult_t AllReduceDdaNvsFlatThreadedAlgo::allReduce() {
  INFO(NCCL_COLL, "AllReduceDdaNvsFlatThreadedAlgo::allReduce");
  NCCLCHECK(NCCL_TYPED_CALL(datatype_, launchKernel));
  return ncclSuccess;
}

} // namespace algorithms
} // namespace nccl
