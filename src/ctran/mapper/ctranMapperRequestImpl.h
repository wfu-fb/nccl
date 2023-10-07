// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_MAPPER_REQUEST_IMPL_H_
#define CTRAN_MAPPER_REQUEST_IMPL_H_

#include "ctranMapper.h"

class ctranMapperRequest::impl {
public:
  impl() = default;
  ~impl() = default;

  enum {
    CTRAN_REQUEST_STATE_UNSET,
    CTRAN_REQUEST_STATE_INCOMPLETE,
    CTRAN_REQUEST_STATE_COMPLETE,
  } state;
};

#endif
