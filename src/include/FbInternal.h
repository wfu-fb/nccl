// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#ifdef ENABLE_FB_INTERNAL
#include "internal.h"
#else

// define wrapper of internal upload API and always return false
#define ncclIsFbPath(path) (false)
#define ncclFbUpload(args...) \
  do {                        \
  } while (0)

#endif
