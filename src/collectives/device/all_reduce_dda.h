// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#if !defined(ALL_REDUCE_DDA_H_INCLUDED)
#define ALL_REDUCE_DDA_H_INCLUDED

#define DECL_DDA_FUNC_NRANKS(T, NRANKS)                                 \
  template __global__ void ncclKernel_AllReduce_DDA_Flat<T, NRANKS>(    \
    uintptr_t barrierFlag,                                              \
    int barrierMboxId,                                                  \
    struct commMd *commMdDev,                                 \
    int rank,                                                           \
    const T* sendbuff,                                                  \
    T* recvbuff,                                                        \
    size_t count);                                                      \
  template __global__ void ncclKernel_AllReduce_DDA_Flat_ipc<T, NRANKS>( \
    uintptr_t barrierFlag,                                              \
    int barrierMboxId,                                                  \
    struct commMd *commMdDev,                                 \
    int rank,                                                           \
    const T* sendbuff,                                                  \
    T* recvbuff,                                                        \
    size_t count);                                                      \
  template __global__ void ncclKernel_AllReduce_DDA_Tree<T, NRANKS>(    \
    uintptr_t barrierFlag,                                              \
    int barrierMboxId,                                                  \
    struct commMd *commMdDev,                                 \
    int rank,                                                           \
    const T* sendbuff,                                                  \
    T* recvbuff,                                                        \
    size_t count);                                                      \
  template __global__ void ncclKernel_AllReduce_DDA_Tree_ipc<T, NRANKS>( \
    uintptr_t barrierFlag,                                              \
    int barrierMboxId,                                                  \
    struct commMd *commMdDev,                                 \
    int rank,                                                           \
    const T* sendbuff,                                                  \
    T* recvbuff,                                                        \
    size_t count);                                                      \
  template __global__ void ncclKernel_AllReduce_DDA_HCM_Flat<T, NRANKS>( \
    uintptr_t barrierFlag,                                              \
    int barrierMboxId,                                                  \
    struct commMd *commMdDev,                                 \
    int rank,                                                           \
    const T* sendbuff,                                                  \
    T* recvbuff,                                                        \
    size_t count);                                                      \
  template __global__ void ncclKernel_AllReduce_DDA_HCM_Tree<T, NRANKS>( \
    uintptr_t barrierFlag,                                              \
    int barrierMboxId,                                                  \
    struct commMd *commMdDev,                                 \
    int rank,                                                           \
    const T* sendbuff,                                                  \
    T* recvbuff,                                                        \
    size_t count);

#define DECL_DDA_FUNC(T)      \
  DECL_DDA_FUNC_NRANKS(T, 2); \
  DECL_DDA_FUNC_NRANKS(T, 4); \
  DECL_DDA_FUNC_NRANKS(T, 8); \
  DECL_DDA_FUNC_NRANKS(T, 16)

#endif /* ALL_REDUCE_DDA_H_INCLUDED */
