#include "Ctran.h"
#include "argcheck.h"
#include "comm.h"
#include "nccl.h"

NCCL_API(
    ncclResult_t,
    ncclAllToAll,
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclAllToAll(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  NCCLCHECK(CudaPtrCheck(sendbuff, comm, "sendbuff", "ncclAllToAll"));
  NCCLCHECK(CudaPtrCheck(recvbuff, comm, "recvbuff", "ncclAllToAll"));
  if (sendbuff == recvbuff) {
    WARN(
        "Found sendbuff %p == recvbuff %p. In-place ncclAllToAll is not supported.",
        sendbuff,
        recvbuff);
    return ncclInvalidArgument;
  }

  // Do nothing if count is 0
  if (count == 0) {
    return ncclSuccess;
  }

  if (ctranAllToAllSupport(count, datatype, comm) &&
      NCCL_ALLTOALL_ALGO == NCCL_ALLTOALL_ALGO::ctran) {
    return ctranAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
  }

  // fallback to default send/recv based alltoallv

  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < comm->nRanks; r++) {
    if (count) {
      NCCLCHECK(ncclSend(
          ((char*)sendbuff) + r * count * ncclTypeSize(datatype),
          count,
          datatype,
          r,
          comm,
          stream));
    }
    if (count) {
      NCCLCHECK(ncclRecv(
          ((char*)recvbuff) + r * count * ncclTypeSize(datatype),
          count,
          datatype,
          r,
          comm,
          stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}
