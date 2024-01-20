// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_ALL_TO_ALLV_IMPL_H_
#define CTRAN_ALL_TO_ALLV_IMPL_H_

#include "CtranMapper.h"
#include "nccl.h"

ncclResult_t ctranAllToAllvIbImpl(
    const void* sendbuff,
    std::vector<size_t>& sendCounts,
    std::vector<size_t>& sDispls,
    void* recvbuff,
    std::vector<size_t>& recvCounts,
    std::vector<size_t>& rDispls,
    ncclDataType_t datatype,
    ncclComm_t comm,
    std::unique_ptr<CtranMapperTimestamp> timestamp);

#endif
