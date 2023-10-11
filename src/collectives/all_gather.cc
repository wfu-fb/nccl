/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "ctranAlgos.h"

NCCL_PARAM(AgDirectCutoff, "AG_DIRECT_CUTOFF", 512 * 1024);
NCCL_PARAM(CtranDisableLocalIb, "CTRAN_DISABLE_LOCAL_IB", 0);

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  int nRanks = comm->nRanks;
  size_t rankOffset = sendcount * ncclTypeSize(datatype);
  bool directSend = (comm->localRanks == 1) && (rankOffset <= ncclParamAgDirectCutoff());
  bool disableLocalIb = (ncclParamCtranDisableLocalIb() && comm->localRanks != 1) ? true : false;

  ctranAlgo algo = ctranAlgoGet(ctranAlgoType::ALLGATHER);

  // only use CTRAN for inter-node only allgather
  if (comm->ctranMapper != nullptr && !disableLocalIb && nRanks > 1 && rankOffset > getpagesize()) {
    if (algo == ctranAlgo::ALLGATHER_CTRAN_DIRECT) {
      return ctranAllGatherDirect(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    } else if (algo == ctranAlgo::ALLGATHER_CTRAN_RING) {
      return ctranAllGatherRing(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    } else if (algo == ctranAlgo::ALLGATHER_CTRAN_RD) {
      return ctranAllGatherRd(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    }
  }

  if (directSend) {
    if (sendcount == 0) return ncclSuccess;

    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < nRanks; r++) {
      NCCLCHECK(ncclSend(
         ((char*)sendbuff), sendcount, datatype, r, comm, stream));
      NCCLCHECK(ncclRecv(
         ((char*)recvbuff) + r * rankOffset, sendcount, datatype, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
  }

  // Just pass the size of one message and not the total bytes sent/received.
  constexpr nvtxPayloadSchemaEntry_t AllGatherSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"}
  };
  size_t msgsize = sendcount * ncclTypeSize(datatype);
  NVTX3_FUNC_WITH_PARAMS(AllGather, AllGatherSchema, msgsize)

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
