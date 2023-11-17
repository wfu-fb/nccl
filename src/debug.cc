/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "nccl_net.h"
#include "param.h"
#include <stdlib.h>
#include <stdarg.h>
#include <sys/syscall.h>
#include <iomanip>
#include <sstream>

int ncclDebugLevel = -1;
static int pid = -1;
static char hostname[1024];
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = ""; // Global string for the last error in human readable form
uint64_t ncclDebugMask = NCCL_INIT|NCCL_ENV; // Default debug sub-system mask is INIT and ENV
FILE *ncclDebugFile = stdout;
pthread_mutex_t ncclDebugLock = PTHREAD_MUTEX_INITIALIZER;
std::chrono::steady_clock::time_point ncclEpoch;

static __thread int tid = -1;

void ncclDebugInit() {
  pthread_mutex_lock(&ncclDebugLock);
  if (ncclDebugLevel != -1) { pthread_mutex_unlock(&ncclDebugLock); return; }
  initEnv();
  const char* nccl_debug = getenv("NCCL_DEBUG");
  int tempNcclDebugLevel = -1;
  if (nccl_debug == NULL) {
    tempNcclDebugLevel = NCCL_LOG_NONE;
  } else if (strcasecmp(nccl_debug, "VERSION") == 0) {
    tempNcclDebugLevel = NCCL_LOG_VERSION;
  } else if (strcasecmp(nccl_debug, "WARN") == 0) {
    tempNcclDebugLevel = NCCL_LOG_WARN;
  } else if (strcasecmp(nccl_debug, "INFO") == 0) {
    tempNcclDebugLevel = NCCL_LOG_INFO;
  } else if (strcasecmp(nccl_debug, "ABORT") == 0) {
    tempNcclDebugLevel = NCCL_LOG_ABORT;
  } else if (strcasecmp(nccl_debug, "TRACE") == 0) {
    tempNcclDebugLevel = NCCL_LOG_TRACE;
  }

  /* Parse the NCCL_DEBUG_SUBSYS env var
   * This can be a comma separated list such as INIT,COLL
   * or ^INIT,COLL etc
   */
  char* ncclDebugSubsysEnv = getenv("NCCL_DEBUG_SUBSYS");
  if (ncclDebugSubsysEnv != NULL) {
    int invert = 0;
    if (ncclDebugSubsysEnv[0] == '^') { invert = 1; ncclDebugSubsysEnv++; }
    ncclDebugMask = invert ? ~0ULL : 0ULL;
    char *ncclDebugSubsys = strdup(ncclDebugSubsysEnv);
    char *subsys = strtok(ncclDebugSubsys, ",");
    while (subsys != NULL) {
      uint64_t mask = 0;
      if (strcasecmp(subsys, "INIT") == 0) {
        mask = NCCL_INIT;
      } else if (strcasecmp(subsys, "COLL") == 0) {
        mask = NCCL_COLL;
      } else if (strcasecmp(subsys, "P2P") == 0) {
        mask = NCCL_P2P;
      } else if (strcasecmp(subsys, "SHM") == 0) {
        mask = NCCL_SHM;
      } else if (strcasecmp(subsys, "NET") == 0) {
        mask = NCCL_NET;
      } else if (strcasecmp(subsys, "GRAPH") == 0) {
        mask = NCCL_GRAPH;
      } else if (strcasecmp(subsys, "TUNING") == 0) {
        mask = NCCL_TUNING;
      } else if (strcasecmp(subsys, "ENV") == 0) {
        mask = NCCL_ENV;
      } else if (strcasecmp(subsys, "ALLOC") == 0) {
        mask = NCCL_ALLOC;
      } else if (strcasecmp(subsys, "CALL") == 0) {
        mask = NCCL_CALL;
      } else if (strcasecmp(subsys, "PROXY") == 0) {
        mask = NCCL_PROXY;
      } else if (strcasecmp(subsys, "NVLS") == 0) {
        mask = NCCL_NVLS;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = NCCL_ALL;
      }
      if (mask) {
        if (invert) ncclDebugMask &= ~mask; else ncclDebugMask |= mask;
      }
      subsys = strtok(NULL, ",");
    }
    free(ncclDebugSubsys);
  }

  // Cache pid and hostname
  getHostName(hostname, 1024, '.');
  pid = getpid();

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NCCL_DEBUG level is > VERSION
   */
  const char* ncclDebugFileEnv = getenv("NCCL_DEBUG_FILE");
  if (tempNcclDebugLevel > NCCL_LOG_VERSION && ncclDebugFileEnv != NULL) {
    int c = 0;
    char debugFn[PATH_MAX+1] = "";
    char *dfn = debugFn;
    while (ncclDebugFileEnv[c] != '\0' && c < PATH_MAX) {
      if (ncclDebugFileEnv[c++] != '%') {
        *dfn++ = ncclDebugFileEnv[c-1];
        continue;
      }
      switch (ncclDebugFileEnv[c++]) {
        case '%': // Double %
          *dfn++ = '%';
          break;
        case 'h': // %h = hostname
          dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
          break;
        case 'p': // %p = pid
          dfn += snprintf(dfn, PATH_MAX, "%d", pid);
          break;
        default: // Echo everything we don't understand
          *dfn++ = '%';
          *dfn++ = ncclDebugFileEnv[c-1];
          break;
      }
    }
    *dfn = '\0';
    if (debugFn[0] != '\0') {
      FILE *file = fopen(debugFn, "w");
      if (file != nullptr) {
        setbuf(file, nullptr); // disable buffering
        ncclDebugFile = file;
      }
    }
  }

  ncclEpoch = std::chrono::steady_clock::now();
  __atomic_store_n(&ncclDebugLevel, tempNcclDebugLevel, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&ncclDebugLock);
}

static std::string getTime(void) {
  auto now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    now.time_since_epoch()) % 1000000;

  std::stringstream timeSs;
  timeSs << std::put_time(std::localtime(&now_c), "%FT%T.") << std::setfill('0')
         << std::setw(6) << now_us.count();
  return timeSs.str();
}

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) {
  if (__atomic_load_n(&ncclDebugLevel, __ATOMIC_ACQUIRE) == -1) ncclDebugInit();
  if (ncclDebugNoWarn != 0 && level == NCCL_LOG_WARN) { level = NCCL_LOG_INFO; flags = ncclDebugNoWarn; }

  // Save the last error (WARN) as a human readable string
  if (level == NCCL_LOG_WARN) {
    pthread_mutex_lock(&ncclDebugLock);
    va_list vargs;
    va_start(vargs, fmt);
    (void) vsnprintf(ncclLastError, sizeof(ncclLastError), fmt, vargs);
    va_end(vargs);
    pthread_mutex_unlock(&ncclDebugLock);
  }
  if (ncclDebugLevel < level || ((flags & ncclDebugMask) == 0)) return;

  if (tid == -1) {
    tid = syscall(SYS_gettid);
  }

  int cudaDev;
  if (!(level == NCCL_LOG_TRACE && flags == NCCL_CALL)) {
    cudaGetDevice(&cudaDev);
  }

  char buffer[1024];
  size_t len = 0;
  auto delta = std::chrono::steady_clock::now() - ncclEpoch;
  std::string timestamp = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(delta).count());
  if (level == NCCL_LOG_WARN) {
    len = snprintf(buffer, sizeof(buffer), "\n%s %s:%d:%d [%d] %s %s:%d NCCL WARN ",
                   getTime().c_str(), hostname, pid, tid, cudaDev, timestamp.c_str(), filefunc, line);
  } else if (level == NCCL_LOG_INFO) {
    len = snprintf(buffer, sizeof(buffer), "%s %s:%d:%d [%d] %s NCCL INFO ", getTime().c_str(), hostname, pid, tid, cudaDev, timestamp.c_str());
  } else if (level == NCCL_LOG_TRACE && flags == NCCL_CALL) {
    len = snprintf(buffer, sizeof(buffer), "%s %s:%d:%d %s NCCL CALL ", getTime().c_str(), hostname, pid, tid, timestamp.c_str());
  } else if (level == NCCL_LOG_TRACE) {
    len = snprintf(buffer, sizeof(buffer), "%s %s:%d:%d [%d] %s %s:%d NCCL TRACE ",
                   getTime().c_str(), hostname, pid, tid, cudaDev, timestamp.c_str(), filefunc, line);
  }

  if (len) {
    va_list vargs;
    va_start(vargs, fmt);
    len += vsnprintf(buffer+len, sizeof(buffer)-len, fmt, vargs);
    va_end(vargs);
    buffer[len++] = '\n';
    fwrite(buffer, 1, len, ncclDebugFile);
  }
}

NCCL_PARAM(SetThreadName, "SET_THREAD_NAME", 0);

void ncclSetThreadName(pthread_t thread, const char *fmt, ...) {
  // pthread_setname_np is nonstandard GNU extension
  // needs the following feature test macro
#ifdef _GNU_SOURCE
  if (ncclParamSetThreadName() != 1) return;
  char threadName[NCCL_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, NCCL_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  pthread_setname_np(thread, threadName);
#endif
}
