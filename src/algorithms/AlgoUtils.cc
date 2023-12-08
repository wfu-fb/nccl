// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "AlgoUtils.h"

#include <cmath>
#include <utility>

#include "checks.h"

namespace nccl {
namespace algorithms {

template <typename T>
ncclResult_t getDataSize_impl(size_t* ret) {
  *ret = sizeof(T);
  return ncclSuccess;
}

size_t getDataSize(ncclDataType_t datatype) {
  size_t dataSize{0};
  NCCLCHECKIGNORE(NCCL_TYPED_CALL(datatype, getDataSize_impl, &dataSize));
  return dataSize;
}

std::pair<dim3, dim3> getGridAndBlockDims(
    const void* func,
    size_t count,
    ncclDataType_t datatype,
    size_t maxBlocks) {
  cudaFuncAttributes funcAttr;
  CUDACHECKIGNORE(cudaFuncGetAttributes(&funcAttr, func));

  unsigned int minBlocks = 1;

  unsigned int minThreads = 32; // warp size
  unsigned int maxThreads = funcAttr.maxThreadsPerBlock;

  const size_t dataSize = getDataSize(datatype);
  const size_t elementsPerThread =
      16 / dataSize; // we do 16 Byte load in kernel

  dim3 grid{1, 1, 1};
  dim3 block{1, 1, 1};

  if (count <= minBlocks * minThreads * elementsPerThread) {
    // for small counts, use the minimum number of blocks and
    // threads, while keeping elementsPerThread elements to be
    // computed by each thread.
    grid.x = minBlocks;
    block.x = minThreads;
  } else if (count < minBlocks * maxThreads * elementsPerThread) {
    // for slightly larger counts, increase the number of threads
    // per block to up to the maximum number of threads.
    grid.x = minBlocks;
    block.x =
        static_cast<int>(std::ceil(count / (minBlocks * elementsPerThread)));
  } else if (count < maxBlocks * maxThreads * elementsPerThread) {
    // for even larger counts, increase the number of blocks to up
    // to the maximum number of blocks.
    grid.x =
        static_cast<int>(std::ceil(count / (maxThreads * elementsPerThread)));
    block.x = maxThreads;
  } else {
    // for even larger counts, use the maximum number of threads
    // and blocks, and let each thread compute more elements.
    grid.x = maxBlocks;
    block.x = maxThreads;
  }

  return std::make_pair(grid, block);
}

} // namespace algorithms
} // namespace nccl
