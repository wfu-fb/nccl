#include "argcheck.h"
#include "comm.h"
#include "nccl.h"

NCCL_API(
    ncclResult_t,
    ncclAllToAllv,
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclAllToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  NCCLCHECK(CudaPtrCheck(sendbuff, comm, "sendbuff", "ncclAllToAllv"));
  NCCLCHECK(CudaPtrCheck(recvbuff, comm, "recvbuff", "ncclAllToAllv"));
  if (sendbuff == recvbuff) {
    WARN(
        "Found sendbuff %p == recvbuff %p. In-place ncclAllToAllv is not supported.",
        sendbuff,
        recvbuff);
    return ncclInvalidArgument;
  }

  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < comm->nRanks; r++) {
    if (sendcounts[r]) {
      NCCLCHECK(ncclSend(
          ((char*)sendbuff) + sdispls[r] * ncclTypeSize(datatype),
          sendcounts[r],
          datatype,
          r,
          comm,
          stream));
    }
    if (recvcounts[r]) {
      NCCLCHECK(ncclRecv(
          ((char*)recvbuff) + rdispls[r] * ncclTypeSize(datatype),
          recvcounts[r],
          datatype,
          r,
          comm,
          stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}
