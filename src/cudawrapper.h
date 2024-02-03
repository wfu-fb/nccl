/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Meta Platforms, Inc. and affiliates. Confidential and proprietary.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CUDAWRAPPER_H_
#define NCCL_CUDAWRAPPER_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include <cstddef>
#include <cstdio>
#include <memory>

#define NCCL_HAS_CUDA_WRAPPER 1

#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>
#else
typedef CUresult (CUDAAPI *PFN_cuInit_v2000)(unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuDriverGetVersion_v2020)(int *driverVersion);
typedef CUresult (CUDAAPI *PFN_cuGetProcAddress_v11030)(const char *symbol, void **pfn, int driverVersion, cuuint64_t flags);
#endif

#define CUPFN(symbol) pfn_##symbol

#define DECLARE_CUDA_PFN_EXTERN(symbol,version) extern PFN_##symbol##_v##version pfn_##symbol

#if CUDART_VERSION >= 11030
/* CUDA Driver functions loaded with cuGetProcAddress for versioning */
DECLARE_CUDA_PFN_EXTERN(cuDeviceGet, 2000);
DECLARE_CUDA_PFN_EXTERN(cuDeviceGetAttribute, 2000);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorString, 6000);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorName, 6000);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAddressRange, 3020);
DECLARE_CUDA_PFN_EXTERN(cuCtxCreate, 3020);
DECLARE_CUDA_PFN_EXTERN(cuCtxDestroy, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxSetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetDevice, 2000);
DECLARE_CUDA_PFN_EXTERN(cuPointerGetAttribute, 4000);
// cuMem API support
DECLARE_CUDA_PFN_EXTERN(cuMemAddressReserve, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemAddressFree, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemCreate, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAllocationGranularity, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemExportToShareableHandle, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemImportFromShareableHandle, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemMap, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemRelease, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemRetainAllocationHandle, 11000);
DECLARE_CUDA_PFN_EXTERN(cuMemSetAccess, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemUnmap, 10020);
#if CUDA_VERSION >= 11070
DECLARE_CUDA_PFN_EXTERN(cuMemGetHandleForAddressRange, 11070); // DMA-BUF support
#endif
#if CUDA_VERSION >= 12010
/* NVSwitch Multicast support */
DECLARE_CUDA_PFN_EXTERN(cuMulticastAddDevice, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindMem, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindAddr, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastCreate, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastGetGranularity, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastUnbind, 12010);
#endif
#endif

/* CUDA Driver functions loaded with dlsym() */
DECLARE_CUDA_PFN_EXTERN(cuInit, 2000);
DECLARE_CUDA_PFN_EXTERN(cuDriverGetVersion, 2020);
DECLARE_CUDA_PFN_EXTERN(cuGetProcAddress, 11030);

class CudaWrapper {
 public:
  bool mock_{false};

  CudaWrapper(bool mock_) : mock_(mock_) {}

  // Driver functions
#if CUDART_VERSION >= 11030
  CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuCtxCreate(pctx, flags, dev));
    }
  }

  CUresult cuCtxGetDevice(CUdevice* device) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuCtxGetDevice(device));
    }
  }

  CUresult cuCtxSetCurrent(CUcontext ctx) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuCtxSetCurrent(ctx));
    }
  }

  CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuDeviceGet(device, ordinal));
    }
  }

  CUresult
  cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuDeviceGetAttribute(pi, attrib, dev));
    }
  }

  CUresult cuDriverGetVersion(int* driverVersion) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuDriverGetVersion(driverVersion));
    }
  }

  CUresult cuGetErrorString(CUresult error, const char** pStr) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuGetErrorString(error, pStr));
    }
  }

  CUresult cuInit(unsigned int Flags) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuInit(Flags));
    }
  }

  CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuMemAddressFree(ptr, size));
    }
  }

  CUresult cuMemAddressReserve(
      CUdeviceptr* ptr,
      size_t size,
      size_t alignment,
      CUdeviceptr addr,
      unsigned long long flags) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuMemAddressReserve(ptr, size, alignment, addr, flags));
    }
  }

  CUresult cuMemCreate(
      CUmemGenericAllocationHandle* handle,
      size_t size,
      const CUmemAllocationProp* prop,
      unsigned long long flags) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuMemCreate(handle, size, prop, flags));
    }
  }

  CUresult cuMemExportToShareableHandle(
      void* shareableHandle,
      CUmemGenericAllocationHandle handle,
      CUmemAllocationHandleType handleType,
      unsigned long long flags) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuMemExportToShareableHandle(
          shareableHandle, handle, handleType, flags));
    }
  }

  CUresult
  cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuMemGetAddressRange(pbase, psize, dptr));
    }
  }

  CUresult cuMemGetAllocationGranularity(
      size_t* granularity,
      const CUmemAllocationProp* prop,
      CUmemAllocationGranularity_flags option) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuMemGetAllocationGranularity(granularity, prop, option));
    }
  }

  CUresult cuMemGetHandleForAddressRange(
      void* handle,
      CUdeviceptr dptr,
      size_t size,
      CUmemRangeHandleType handleType,
      unsigned long long flags) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(
          cuMemGetHandleForAddressRange(handle, dptr, size, handleType, flags));
    }
  }

  CUresult cuMemImportFromShareableHandle(
      CUmemGenericAllocationHandle* handle,
      void* osHandle,
      CUmemAllocationHandleType shHandleType) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(
          cuMemImportFromShareableHandle(handle, osHandle, shHandleType));
    }
  }

  CUresult cuMemMap(
      CUdeviceptr ptr,
      size_t size,
      size_t offset,
      CUmemGenericAllocationHandle handle,
      unsigned long long flags) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuMemMap(ptr, size, offset, handle, flags));
    }
  }

  CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuMemRelease(handle));
    }
  }

  CUresult cuMemRetainAllocationHandle(
      CUmemGenericAllocationHandle* handle,
      void* addr) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuMemRetainAllocationHandle(handle, addr));
    }
  }

  CUresult cuMemSetAccess(
      CUdeviceptr ptr,
      size_t size,
      const CUmemAccessDesc* desc,
      size_t count) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuMemSetAccess(ptr, size, desc, count));
    }
  }

  CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuMemUnmap(ptr, size));
    }
  }
#endif

#if CUDA_VERSION >= 12010
  CUresult cuMulticastAddDevice(
      CUmemGenericAllocationHandle mcHandle,
      CUdevice dev) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return ::cuMulticastAddDevice(mcHandle, dev);
    }
  }

  CUresult cuMulticastBindAddr(
      CUmemGenericAllocationHandle mcHandle,
      size_t mcOffset,
      CUdeviceptr memptr,
      size_t size,
      unsigned long long flags) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return ::cuMulticastBindAddr(mcHandle, mcOffset, memptr, size, flags);
    }
  }

  CUresult cuMulticastBindMem(
      CUmemGenericAllocationHandle mcHandle,
      size_t mcOffset,
      CUmemGenericAllocationHandle memHandle,
      size_t memOffset,
      size_t size,
      unsigned long long flags) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return ::cuMulticastBindMem(
          mcHandle, mcOffset, memHandle, memOffset, size, flags);
    }
  }

  CUresult cuMulticastCreate(
      CUmemGenericAllocationHandle* mcHandle,
      const CUmulticastObjectProp* prop) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return ::cuMulticastCreate(mcHandle, prop);
    }
  }

  CUresult cuMulticastGetGranularity(
      size_t* granularity,
      const CUmulticastObjectProp* prop,
      CUmulticastGranularity_flags option) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return ::cuMulticastGetGranularity(granularity, prop, option);
    }
  }

  CUresult cuMulticastUnbind(
      CUmemGenericAllocationHandle mcHandle,
      CUdevice dev,
      size_t mcOffset,
      size_t size) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return ::cuMulticastUnbind(mcHandle, dev, mcOffset, size);
    }
  }
#endif

  CUresult cuPointerGetAttribute(
      void* data,
      CUpointer_attribute attribute,
      CUdeviceptr ptr) {
    if (mock_) {
      return CUDA_SUCCESS;
    } else {
      return CUPFN(cuPointerGetAttribute(data, attribute, ptr));
    }
  }

  // Runtime API functions
  cudaError_t
  cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
    }
  }

  cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaDeviceEnablePeerAccess(peerDevice, flags);
    }
  }

  cudaError_t
  cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaDeviceGetAttribute(value, attr, device);
    }
  }

  cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaDeviceGetPCIBusId(pciBusId, len, device);
    }
  }

  cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaDeviceSetLimit(limit, value);
    }
  }

  cudaError_t cudaDeviceSynchronize(void) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaDeviceSynchronize();
    }
  }

  cudaError_t cudaDriverGetVersion(int* driverVersion) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaDriverGetVersion(driverVersion);
    }
  }

  cudaError_t cudaEventCreate(cudaEvent_t* event) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaEventCreate(event);
    }
  }

  cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaEventCreateWithFlags(event, flags);
    }
  }

  cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaEventDestroy(event);
    }
  }

  cudaError_t cudaEventQuery(cudaEvent_t event) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaEventQuery(event);
    }
  }

  cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaEventRecord(event, stream);
    }
  }

  cudaError_t cudaFree(void* devPtr) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaFree(devPtr);
    }
  }

  cudaError_t cudaFreeHost(void* ptr) {
    if (mock_) {
      free(ptr);
      return cudaSuccess;
    } else {
      return ::cudaFreeHost(ptr);
    }
  }

  cudaError_t cudaFuncGetAttributes(
      struct cudaFuncAttributes* attr,
      const void* func) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaFuncGetAttributes(attr, func);
    }
  }

  cudaError_t cudaFuncSetAttribute(
      const void* func,
      enum cudaFuncAttribute attr,
      int value) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaFuncSetAttribute(func, attr, value);
    }
  }

  cudaError_t cudaGetDevice(int* device) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaGetDevice(device);
    }
  }

  cudaError_t cudaGetDeviceCount(int* count) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaGetDeviceCount(count);
    }
  }

  cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaGetDeviceProperties(prop, device);
    }
  }

  cudaError_t cudaGetDriverEntryPoint(
      const char* symbol,
      void** funcPtr,
      unsigned long long flags,
      enum cudaDriverEntryPointQueryResult* driverStatus = nullptr) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus);
    }
  }

  const char* cudaGetErrorString(cudaError_t error) {
    if (mock_) {
      return "";
    } else {
      return ::cudaGetErrorString(error);
    }
  }

  cudaError_t cudaGetLastError(void) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaGetLastError();
    }
  }

  cudaError_t cudaGraphAddEventRecordNode(
      cudaGraphNode_t* pGraphNode,
      cudaGraph_t graph,
      const cudaGraphNode_t* pDependencies,
      size_t numDependencies,
      cudaEvent_t event) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaGraphAddEventRecordNode(
          pGraphNode, graph, pDependencies, numDependencies, event);
    }
  }

  cudaError_t cudaGraphAddEventWaitNode(
      cudaGraphNode_t* pGraphNode,
      cudaGraph_t graph,
      const cudaGraphNode_t* pDependencies,
      size_t numDependencies,
      cudaEvent_t event) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaGraphAddEventWaitNode(
          pGraphNode, graph, pDependencies, numDependencies, event);
    }
  }

  cudaError_t cudaGraphAddHostNode(
      cudaGraphNode_t* pGraphNode,
      cudaGraph_t graph,
      const cudaGraphNode_t* pDependencies,
      size_t numDependencies,
      const struct cudaHostNodeParams* pNodeParams) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaGraphAddHostNode(
          pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
    }
  }

  cudaError_t cudaGraphAddKernelNode(
      cudaGraphNode_t* pGraphNode,
      cudaGraph_t graph,
      const cudaGraphNode_t* pDependencies,
      size_t numDependencies,
      const struct cudaKernelNodeParams* pNodeParams) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaGraphAddKernelNode(
          pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
    }
  }

  cudaError_t cudaGraphRetainUserObject(
      cudaGraph_t graph,
      cudaUserObject_t object,
      unsigned int count = 1,
      unsigned int flags = 0) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaGraphRetainUserObject(graph, object, count, flags);
    }
  }

  cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
    if (mock_) {
      *pHost = malloc(size);
      return cudaSuccess;
    } else {
      return ::cudaHostAlloc(pHost, size, flags);
    }
  }

  cudaError_t
  cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaHostGetDevicePointer(pDevice, pHost, flags);
    }
  }

  cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaHostRegister(ptr, size, flags);
    }
  }

  cudaError_t CUDARTAPI cudaHostUnregister(void* ptr) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaHostUnregister(ptr);
    }
  }

  cudaError_t cudaIpcCloseMemHandle(void* devPtr) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaIpcCloseMemHandle(devPtr);
    }
  }

  cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaIpcGetMemHandle(handle, devPtr);
    }
  }

  cudaError_t cudaIpcOpenMemHandle(
      void** devPtr,
      cudaIpcMemHandle_t handle,
      unsigned int flags) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaIpcOpenMemHandle(devPtr, handle, flags);
    }
  }

  cudaError_t
  cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaLaunchHostFunc(stream, fn, userData);
    }
  }

  cudaError_t cudaLaunchKernel(
      const void* func,
      dim3 gridDim,
      dim3 blockDim,
      void** args,
      size_t sharedMem,
      cudaStream_t stream) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaLaunchKernel(
          func, gridDim, blockDim, args, sharedMem, stream);
    }
  }

  cudaError_t CUDARTAPI cudaLaunchKernelExC(
      const cudaLaunchConfig_t* config,
      const void* func,
      void** args) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaLaunchKernelExC(config, func, args);
    }
  }

  cudaError_t cudaMalloc(void** devPtr, std::size_t size) {
    if (mock_) {
      *devPtr = malloc(size);
      return cudaSuccess;
    } else {
      return ::cudaMalloc(devPtr, size);
    }
  }

  cudaError_t cudaMemcpy(
      void* dst,
      const void* src,
      size_t count,
      enum cudaMemcpyKind kind) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaMemcpy(dst, src, count, kind);
    }
  }

  cudaError_t cudaMemcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      enum cudaMemcpyKind kind,
      cudaStream_t stream = 0) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaMemcpyAsync(dst, src, count, kind, stream);
    }
  }

  cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaMemset(devPtr, value, count);
    }
  }

  cudaError_t cudaMemsetAsync(
      void* devPtr,
      int value,
      size_t count,
      cudaStream_t stream = 0) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaMemsetAsync(devPtr, value, count, stream);
    }
  }

  cudaError_t cudaPointerGetAttributes(
      struct cudaPointerAttributes* attributes,
      const void* ptr) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaPointerGetAttributes(attributes, ptr);
    }
  }

  cudaError_t cudaSetDevice(int device) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaSetDevice(device);
    }
  }

  cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaStreamCreate(pStream);
    }
  }

  cudaError_t cudaStreamCreateWithFlags(
      cudaStream_t* pStream,
      unsigned int flags) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaStreamCreateWithFlags(pStream, flags);
    }
  }

  cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaStreamDestroy(stream);
    }
  }

  cudaError_t cudaStreamGetCaptureInfo(
      cudaStream_t stream,
      enum cudaStreamCaptureStatus* captureStatus_out,
      unsigned long long* id_out = 0,
      cudaGraph_t* graph_out = 0,
      const cudaGraphNode_t** dependencies_out = 0,
      size_t* numDependencies_out = 0) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaStreamGetCaptureInfo(
          stream,
          captureStatus_out,
          id_out,
          graph_out,
          dependencies_out,
          numDependencies_out);
    }
  }

  cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaStreamSynchronize(stream);
    }
  }

  cudaError_t cudaStreamUpdateCaptureDependencies(
      cudaStream_t stream,
      cudaGraphNode_t* dependencies,
      size_t numDependencies,
      unsigned int flags = 0) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaStreamUpdateCaptureDependencies(
          stream, dependencies, numDependencies, flags);
    }
  }

  cudaError_t cudaStreamWaitEvent(
      cudaStream_t stream,
      cudaEvent_t event,
      unsigned int flags = 0) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaStreamWaitEvent(stream, event, flags);
    }
  }

  cudaError_t cudaThreadExchangeStreamCaptureMode(
      enum cudaStreamCaptureMode* mode) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaThreadExchangeStreamCaptureMode(mode);
    }
  }

  cudaError_t cudaUserObjectCreate(
      cudaUserObject_t* object_out,
      void* ptr,
      cudaHostFn_t destroy,
      unsigned int initialRefcount,
      unsigned int flags) {
    if (mock_) {
      return cudaSuccess;
    } else {
      return ::cudaUserObjectCreate(
          object_out, ptr, destroy, initialRefcount, flags);
    }
  }
};

extern std::shared_ptr<CudaWrapper> cudaWrapper;

#endif
