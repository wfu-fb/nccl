#ifndef NCCL_DATA_EXPORT_H_
#define NCCL_DATA_EXPORT_H_
#include <string>
#include "comm.h"
#include "nccl.h"

#ifdef ENABLE_FB_DATA_EXPORT
ncclResult_t ncclDataExport(
    const void* sendbuff,
    std::size_t nBytes,
    cudaStream_t stream,
    ncclComm_t comm,
    int dest,
    int datatype);
#else
#define ncclDataExport(args...) (ncclSuccess)
#endif // end of ENABLE_FB_DATA_EXPORT

#endif
