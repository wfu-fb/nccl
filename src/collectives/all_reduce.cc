/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "nccl.h"

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  if (comm->algoMgr) {
    // try to get meta customized algo
    auto algo = comm->algoMgr->getAllReduceAlgo(sendbuff, recvbuff, count, datatype, op, comm, stream);
    if (algo) {
      return algo->allReduce();
    }
  }

  ncclDDAAllReduceAlgo_t allreduceAlgo = getAllReduceAlgo(sendbuff, recvbuff, count, datatype, op, comm);
  if (allreduceAlgo != NCCL_DDA_ALLREDUCE_ALGO_DEFAULT) {
    auto ret = ncclAllReduceDDA(sendbuff, recvbuff, count, datatype, op, comm, stream);
    if (ret != ncclInvalidUsage) {
      // return immediately if result is non-ncclInvalidUsage error
      return ret;
    }
  }

  struct NvtxParamsAllReduce {
    size_t bytes;
    ncclRedOp_t op;
  };
  // Just pass the size of one message and not the total bytes sent/received.
  static constexpr nvtxPayloadSchemaEntry_t AllReduceSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsAllReduce, op)}
  };
  NvtxParamsAllReduce payload{count * ncclTypeSize(datatype), op};
  NVTX3_FUNC_WITH_PARAMS(AllReduce, AllReduceSchema, payload)

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
