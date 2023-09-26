// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#if !defined(ALL_REDUCE_DDA_H_INCLUDED)
#define ALL_REDUCE_DDA_H_INCLUDED

#define DECL_DDA_FUNC_NRANKS(T, NRANKS)                                  \
  template __global__ void ncclKernel_AllReduce_DDA_Flat<T, NRANKS>(     \
      uintptr_t * barrierMbox,                                                \
      uintptr_t barrierFlag,                                                  \
      int rank,                                                               \
      const T* sendbuff,                                                      \
      T* recvbuff,                                                            \
      size_t count);                                                          \
  template __global__ void ncclKernel_AllReduce_DDA_Flat_ipc<T, NRANKS>( \
      uintptr_t * barrierMbox,                                                \
      uintptr_t barrierFlag,                                                  \
      int rank,                                                               \
      T* recvbuff,                                                            \
      size_t count,                                                           \
      const T** allSendBuffs);                                                 \
  template __global__ void ncclKernel_AllReduce_DDA_Tree<T, NRANKS>(     \
      uintptr_t * barrierMbox,                                                \
      uintptr_t barrierFlag,                                                  \
      int rank,                                                               \
      const T* sendbuff,                                                      \
      T* recvbuff,                                                            \
      size_t count);                                                          \
  template __global__ void ncclKernel_AllReduce_DDA_Tree_ipc<T, NRANKS>( \
      uintptr_t * barrierMbox,                                                \
      uintptr_t barrierFlag,                                                  \
      int rank,                                                               \
      T** allSendBuffs,                                                       \
      T* recvbuff,                                                            \
      size_t count);                                                          \
  template __global__ void ncclKernel_AllReduce_DDA_HCM_Flat<T, NRANKS>( \
      uintptr_t * cliqueBarrierMbox,                                          \
      uintptr_t * localMbox,                                                  \
      uintptr_t * peerMbox,                                                   \
      uintptr_t barrierFlag,                                                  \
      int cliqueRank,                                                         \
      const T* sendbuff,                                                      \
      T* tmpbuff,                                                             \
      T* recvbuff,                                                            \
      size_t count);                                                          \
  template __global__ void ncclKernel_AllReduce_DDA_HCM_Tree<T, NRANKS>( \
      uintptr_t * cliqueBarrierMbox,                                          \
      uintptr_t * localMbox,                                                  \
      uintptr_t * peerMbox,                                                   \
      uintptr_t barrierFlag,                                                  \
      int cliqueRank,                                                         \
      const T* sendbuff,                                                      \
      T* tmpbuff,                                                             \
      T* recvbuff,                                                            \
      size_t count)

#define DECL_DDA_FUNC(T)      \
  DECL_DDA_FUNC_NRANKS(T, 2); \
  DECL_DDA_FUNC_NRANKS(T, 4); \
  DECL_DDA_FUNC_NRANKS(T, 8); \
  DECL_DDA_FUNC_NRANKS(T, 16)

#endif /* ALL_REDUCE_DDA_H_INCLUDED */
