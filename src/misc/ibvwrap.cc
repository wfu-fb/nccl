/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ibvwrap.h"
#include <sys/types.h>
#include <unistd.h>

#include "ibvsymbols.h"
#include "core.h"

// Explicitly include NTRACE after verbs type definition
// to pass these types to ntrace_rt.h.
#include "ntrace_profiler.h"

static pthread_once_t initOnceControl = PTHREAD_ONCE_INIT;
static ncclResult_t initResult;
struct ncclIbvSymbols ibvSymbols;

ncclResult_t IbvWrapper::wrap_ibv_symbols(void) {
  if (mock_) {
    return ncclSuccess;
  }
  pthread_once(&initOnceControl,
               [](){ initResult = buildIbvSymbols(&ibvSymbols); });
  return initResult;
}

/* CHECK_NOT_NULL: helper macro to check for NULL symbol */
#define CHECK_NOT_NULL(container, internal_name) \
  if (container.internal_name == NULL) { \
     WARN("lib wrapper not initialized."); \
     return ncclInternalError; \
  }

#define IBV_PTR_CHECK_ERRNO(container, internal_name, call, retval, error_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  retval = container.call; \
  if (retval == error_retval) { \
    WARN("Call to " name " failed with error %s", strerror(errno)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_PTR_CHECK(container, internal_name, call, retval, error_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  retval = container.call; \
  if (retval == error_retval) { \
    WARN("Call to " name " failed"); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_INT_CHECK_RET_ERRNO_OPTIONAL(container, internal_name, call, success_retval, name, supported) \
  if (container.internal_name == NULL) { \
    INFO(NCCL_NET, "Call to " name " skipped, internal_name doesn't exist"); \
    *supported = 0; \
    return ncclSuccess; \
  } \
  int ret = container.call; \
  if (ret == ENOTSUP || ret == EOPNOTSUPP) { \
    INFO(NCCL_NET, "Call to " name " failed with error %s errno %d", strerror(ret), ret); \
    *supported = 0; \
    return ncclSuccess; \
  } else if (ret != success_retval) { \
    WARN("Call to " name " failed with error %s errno %d", strerror(ret), ret); \
    *supported = 1; \
    return ncclSystemError; \
  } \
  *supported = 1; \
  return ncclSuccess;

#define IBV_INT_CHECK_RET_ERRNO(container, internal_name, call, success_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  int ret = container.call; \
  if (ret != success_retval) { \
    WARN("Call to " name " failed with error %s errno %d", strerror(ret), ret); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_INT_CHECK(container, internal_name, call, error_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  int ret = container.call; \
  if (ret == error_retval) { \
    WARN("Call to " name " failed"); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

// _NO_RETURN version of the above check macros to enable ntrace post event recording.
// Compare to the original version, it sets ncclret without return.
#define IBV_PTR_CHECK_NO_RETURN(container, internal_name, call, retval, error_retval, name, ncclret) \
  CHECK_NOT_NULL(container, internal_name);                           \
  retval = container.call;                                            \
  if (retval == error_retval) {                                       \
    WARN("Call to " name " failed");                                  \
    ncclret = ncclSystemError;                                        \
  }

#define IBV_INT_CHECK_RET_ERRNO_NO_RETURN(container, internal_name, call, success_retval, name, ncclret) \
  CHECK_NOT_NULL(container, internal_name);                           \
  int ret = container.call;                                           \
  if (ret != success_retval) {                                        \
    WARN("Call to " name " failed with error %s", strerror(ret));     \
    ncclret = ncclSystemError;                                        \
  }

#define IBV_INT_CHECK_NO_RETURN(container, internal_name, call, error_retval, name, ncclret) \
  CHECK_NOT_NULL(container, internal_name); \
  int ret = container.call;                 \
  if (ret == error_retval) {                \
    WARN("Call to " name " failed");        \
    ncclret = ncclSystemError;              \
  }

#define IBV_PASSTHRU(container, internal_name, call) \
  CHECK_NOT_NULL(container, internal_name); \
  container.call; \
  return ncclSuccess;

ncclResult_t IbvWrapper::wrap_ibv_fork_init() {
  if (mock_) {
    return ncclSuccess;
  }
  IBV_INT_CHECK(ibvSymbols, ibv_internal_fork_init, ibv_internal_fork_init(), -1, "ibv_fork_init");
}

ncclResult_t IbvWrapper::wrap_ibv_get_device_list(struct ibv_device ***ret, int *num_devices) {
  if (mock_) {
    return ncclSuccess;
  }
  *ret = ibvSymbols.ibv_internal_get_device_list(num_devices);
  if (*ret == NULL) *num_devices = 0;
  return ncclSuccess;
}

ncclResult_t IbvWrapper::wrap_ibv_free_device_list(struct ibv_device **list) {
  if (mock_) {
    return ncclSuccess;
  }
  IBV_PASSTHRU(ibvSymbols, ibv_internal_free_device_list, ibv_internal_free_device_list(list));
}

const char *IbvWrapper::wrap_ibv_get_device_name(struct ibv_device *device) {
  if (mock_) {
    return "";
  }
  if (ibvSymbols.ibv_internal_get_device_name == NULL) {
    WARN("lib wrapper not initialized.");
    exit(-1);
  }
  return ibvSymbols.ibv_internal_get_device_name(device);
}

ncclResult_t IbvWrapper::wrap_ibv_open_device(struct ibv_context **ret, struct ibv_device *device) { /*returns 0 on success, -1 on failure*/
  if (mock_) {
    return ncclSuccess;
  }
  ncclResult_t ncclret = ncclSuccess;
  IBV_PTR_CHECK_NO_RETURN(ibvSymbols, ibv_internal_open_device, ibv_internal_open_device(device), *ret, NULL, "ibv_open_device", ncclret);
  NTRACE_PROFILING_RECORD(IbvOpenDevice, *ret, device);
  return ncclret;
}

ncclResult_t IbvWrapper::wrap_ibv_close_device(struct ibv_context *context) { /*returns 0 on success, -1 on failure*/
  if (mock_) {
    return ncclSuccess;
  }
  ncclResult_t ncclret = ncclSuccess;
  NTRACE_PROFILING_RECORD(IbvCloseDevice, context);
  IBV_INT_CHECK_NO_RETURN(ibvSymbols, ibv_internal_close_device, ibv_internal_close_device(context), -1, "ibv_close_device", ncclret);
  return ncclret;
}

ncclResult_t IbvWrapper::wrap_ibv_get_async_event(struct ibv_context *context, struct ibv_async_event *event) { /*returns 0 on success, and -1 on error*/
if (mock_) {
    return ncclSuccess;
  }
  IBV_INT_CHECK(ibvSymbols, ibv_internal_get_async_event, ibv_internal_get_async_event(context, event), -1, "ibv_get_async_event");
}

ncclResult_t IbvWrapper::wrap_ibv_ack_async_event(struct ibv_async_event *event) {
  if (mock_) {
    return ncclSuccess;
  }
  IBV_PASSTHRU(ibvSymbols, ibv_internal_ack_async_event, ibv_internal_ack_async_event(event));
}

ncclResult_t IbvWrapper::wrap_ibv_query_device(struct ibv_context *context, struct ibv_device_attr *device_attr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
if (mock_) {
    return ncclSuccess;
  }
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_query_device, ibv_internal_query_device(context, device_attr), 0, "ibv_query_device");
}

ncclResult_t IbvWrapper::wrap_ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  if (mock_) {
    return ncclSuccess;
  }
  ncclResult_t ncclret = ncclSuccess;
  IBV_INT_CHECK_RET_ERRNO_NO_RETURN(ibvSymbols, ibv_internal_query_port, ibv_internal_query_port(context, port_num, port_attr), 0, "ibv_query_port", ncclret);
  NTRACE_PROFILING_RECORD(IbvQueryPort, context, port_num, *port_attr);
  return ncclret;
}

ncclResult_t IbvWrapper::wrap_ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid) {
  if (mock_) {
    return ncclSuccess;
  }
   ncclResult_t ncclret = ncclSuccess;
   IBV_INT_CHECK_RET_ERRNO_NO_RETURN(ibvSymbols, ibv_internal_query_gid, ibv_internal_query_gid(context, port_num, index, gid), 0, "ibv_query_gid", ncclret);
   NTRACE_PROFILING_RECORD(IbvQueryGid, context, port_num, index, *gid);
   return ncclret;
}

ncclResult_t IbvWrapper::wrap_ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr) {
  if (mock_) {
    return ncclSuccess;
  }
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_query_qp, ibv_internal_query_qp(qp, attr, attr_mask, init_attr), 0, "ibv_query_qp");
}

ncclResult_t IbvWrapper::wrap_ibv_alloc_pd(struct ibv_pd **ret, struct ibv_context *context) {
  if (mock_) {
    return ncclSuccess;
  }
  IBV_PTR_CHECK_ERRNO(ibvSymbols, ibv_internal_alloc_pd, ibv_internal_alloc_pd(context), *ret, NULL, "ibv_alloc_pd");
}

ncclResult_t IbvWrapper::wrap_ibv_dealloc_pd(struct ibv_pd *pd) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
if (mock_) {
    return ncclSuccess;
  }
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_dealloc_pd, ibv_internal_dealloc_pd(pd), 0, "ibv_dealloc_pd");
}

ncclResult_t IbvWrapper::wrap_ibv_reg_mr(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, int access) {
  if (mock_) {
    return ncclSuccess;
  }
  IBV_PTR_CHECK_ERRNO(ibvSymbols, ibv_internal_reg_mr, ibv_internal_reg_mr(pd, addr, length, access), *ret, NULL, "ibv_reg_mr");
}

struct ibv_mr * IbvWrapper::wrap_direct_ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access) {
  if (mock_) {
    return nullptr;
  }
  if (ibvSymbols.ibv_internal_reg_mr == NULL) {
    WARN("lib wrapper not initialized.");
    return NULL;
  }
  return ibvSymbols.ibv_internal_reg_mr(pd, addr, length, access);
}

ncclResult_t IbvWrapper::wrap_ibv_reg_mr_iova2(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, int access) {
  if (mock_) {
    return ncclSuccess;
  }
  if (ibvSymbols.ibv_internal_reg_mr_iova2 == NULL) {
    return ncclInternalError;
  }
  if (ret == NULL) { return ncclSuccess; } // Assume dummy call
  IBV_PTR_CHECK_ERRNO(ibvSymbols, ibv_internal_reg_mr_iova2, ibv_internal_reg_mr_iova2(pd, addr, length, iova, access), *ret, NULL, "ibv_reg_mr_iova2");
}

/* DMA-BUF support */
ncclResult_t IbvWrapper::wrap_ibv_reg_dmabuf_mr(struct ibv_mr **ret, struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access) {
  if (mock_) {
    return ncclSuccess;
  }
  IBV_PTR_CHECK_ERRNO(ibvSymbols, ibv_internal_reg_dmabuf_mr, ibv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access), *ret, NULL, "ibv_reg_dmabuf_mr");
}

struct ibv_mr * IbvWrapper::wrap_direct_ibv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access) {
  if (mock_) {
    return nullptr;
  }
  if (ibvSymbols.ibv_internal_reg_dmabuf_mr == NULL) {
    errno = EOPNOTSUPP; // ncclIbDmaBufSupport() requires this errno being set
    return NULL;
  }
  return ibvSymbols.ibv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access);
}

ncclResult_t IbvWrapper::wrap_ibv_dereg_mr(struct ibv_mr *mr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
if (mock_) {
    return ncclSuccess;
  }
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_dereg_mr, ibv_internal_dereg_mr(mr), 0, "ibv_dereg_mr");
}

ncclResult_t IbvWrapper::wrap_ibv_create_comp_channel(struct ibv_comp_channel **ret, struct ibv_context *context) {
  return ncclInternalError;
}

ncclResult_t IbvWrapper::wrap_ibv_destroy_comp_channel(struct ibv_comp_channel *channel) {
  return ncclInternalError;
}

ncclResult_t IbvWrapper::wrap_ibv_create_cq(struct ibv_cq **ret, struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector) {
  if (mock_) {
    return ncclSuccess;
  }
  IBV_PTR_CHECK_ERRNO(ibvSymbols, ibv_internal_create_cq, ibv_internal_create_cq(context, cqe, cq_context, channel, comp_vector), *ret, NULL, "ibv_create_cq");
}

ncclResult_t IbvWrapper::wrap_ibv_destroy_cq(struct ibv_cq *cq) {
  if (mock_) {
    return ncclSuccess;
  }
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_destroy_cq, ibv_internal_destroy_cq(cq), 0, "ibv_destroy_cq");
}

ncclResult_t IbvWrapper::wrap_ibv_destroy_qp(struct ibv_qp *qp) {
  if (mock_) {
    return ncclSuccess;
  }
  ncclResult_t ncclret = ncclSuccess;
  NTRACE_PROFILING_RECORD(IbvDestroyQp, qp);
  IBV_INT_CHECK_RET_ERRNO_NO_RETURN(ibvSymbols, ibv_internal_destroy_qp, ibv_internal_destroy_qp(qp), 0, "ibv_destroy_qp", ncclret);
  return ncclret;
}

ncclResult_t IbvWrapper::wrap_ibv_create_qp(struct ibv_qp **ret, struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr) {
  if (mock_) {
    return ncclSuccess;
  }
  ncclResult_t ncclret = ncclSuccess;
  IBV_PTR_CHECK_NO_RETURN(ibvSymbols, ibv_internal_create_qp, ibv_internal_create_qp(pd, qp_init_attr), *ret, NULL, "ibv_create_qp", ncclret);
  NTRACE_PROFILING_RECORD(IbvCreateQp, *ret, qp_init_attr);
  return ncclret;
}

ncclResult_t IbvWrapper::wrap_ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  if (mock_) {
    return ncclSuccess;
  }
  ncclResult_t ncclret = ncclSuccess;
  NTRACE_PROFILING_RECORD(IbvModifyQp, qp, attr, attr_mask);
  IBV_INT_CHECK_RET_ERRNO_NO_RETURN(ibvSymbols, ibv_internal_modify_qp, ibv_internal_modify_qp(qp, attr, attr_mask), 0, "ibv_modify_qp", ncclret);
  return ncclret;
}

ncclResult_t IbvWrapper::wrap_ibv_query_ece(struct ibv_qp *qp, struct ibv_ece *ece, int* supported) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
if (mock_) {
    return ncclSuccess;
  }
  IBV_INT_CHECK_RET_ERRNO_OPTIONAL(ibvSymbols, ibv_internal_query_ece, ibv_internal_query_ece(qp, ece), 0, "ibv_query_ece", supported);
}

ncclResult_t IbvWrapper::wrap_ibv_set_ece(struct ibv_qp *qp, struct ibv_ece *ece, int* supported) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
if (mock_) {
    return ncclSuccess;
  }
  IBV_INT_CHECK_RET_ERRNO_OPTIONAL(ibvSymbols, ibv_internal_set_ece, ibv_internal_set_ece(qp, ece), 0, "ibv_set_ece", supported);
}

ncclResult_t IbvWrapper::wrap_ibv_event_type_str(char **ret, enum ibv_event_type event) {
  *ret = (char *) ibvSymbols.ibv_internal_event_type_str(event);
  return ncclSuccess;
}

ncclResult_t IbvWrapper::wrap_ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc, int* num_done) {
  int done = cq->context->ops.poll_cq(cq, num_entries, wc); /*returns the number of wcs or 0 on success, a negative number otherwise*/
  NTRACE_PROFILING_RECORD(IbvPollCq, cq, num_entries, wc, done);
  if (done < 0) {
    WARN("Call to ibv_poll_cq() returned %d", done);
    return ncclSystemError;
  }
  *num_done = done;
  return ncclSuccess;
}

ncclResult_t IbvWrapper::wrap_ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr) {
  if (mock_) {
    return ncclSuccess;
  }
  NTRACE_PROFILING_RECORD(IbvPostSend, qp, wr, (const struct ibv_send_wr **)bad_wr, NULL);
  int ret = qp->context->ops.post_send(qp, wr, bad_wr); /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  if (ret != IBV_SUCCESS) {
    WARN("ibv_post_send() failed with error %s, Bad WR %p, First WR %p", strerror(ret), wr, *bad_wr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t IbvWrapper::wrap_ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr) {
  if (mock_) {
    return ncclSuccess;
  }
  NTRACE_PROFILING_RECORD(IbvPostRecv, qp, wr, (const struct ibv_recv_wr **)bad_wr, NULL);
  int ret = qp->context->ops.post_recv(qp, wr, bad_wr); /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  if (ret != IBV_SUCCESS) {
    WARN("ibv_post_recv() failed with error %s", strerror(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}
