// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "stdlib.h"
#include "string.h"
#include "ctranAlgos.h"

ctranAlgo ctranAlgoGet(ctranAlgoType type) {
  switch (type) {
    case ctranAlgoType::ALLGATHER:
      {
        const char *allgatherAlgoStr = getenv("NCCL_ALLGATHER_ALGO");
        if (allgatherAlgoStr != nullptr) {
          if (!strcmp(allgatherAlgoStr, "ctran:direct")) {
            return ctranAlgo::ALLGATHER_CTRAN_DIRECT;
          } else if (!strcmp(allgatherAlgoStr, "ctran:ring")) {
            return ctranAlgo::ALLGATHER_CTRAN_RING;
          }
        }
        return ctranAlgo::ALLGATHER_ORIG;
      }

    default:
      return ctranAlgo::UNKNOWN;
  }
}
