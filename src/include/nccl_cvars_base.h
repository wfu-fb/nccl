// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_CVARS_BASE_H_INCLUDED
#define NCCL_CVARS_BASE_H_INCLUDED

#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include "debug.h"
#include "checks.h"

// Cvar internal logger
// We need avoid calling into default logger because it may call ncclGetEnv() on
// demand in ncclDebugInit() and cause circular call & deadlock. Since CVAR_WARN
// happens usually only at initialization time and is for warning only, we might
// be OK to use separate logger here and always print to stdout.
static int pid = getpid();
static thread_local int tid = syscall(SYS_gettid);
static char hostname[HOST_NAME_MAX];
static bool enableCvarWarn = true;
static int cudaDev = -1;

#define CVAR_WARN(fmt, ...)                                 \
  if (enableCvarWarn) {                                     \
    printf(                                                 \
        "%s %s:%d:%d [%d] %s:%d NCCL WARN CVAR: " fmt "\n", \
        getTime().c_str(),                                  \
        hostname,                                           \
        pid,                                                \
        tid,                                                \
        cudaDev,                                            \
        __FUNCTION__,                                       \
        __LINE__,                                           \
        ##__VA_ARGS__);                                     \
  }

#define CVAR_WARN_UNKNOWN_VALUE(name, value)               \
  do {                                                     \
    CVAR_WARN("Unknown value %s for env %s", value, name); \
  } while (0)

static void initCvarLogger() {
  const char* nccl_debug = getenv("NCCL_DEBUG");
  if (nccl_debug == NULL || strcasecmp(nccl_debug, "VERSION") == 0) {
    enableCvarWarn = false;
  }
  getHostName(hostname, HOST_NAME_MAX, '.');

  // Used for ncclCvarInit time warning only
  CUDACHECKIGNORE(cudaGetDevice(&cudaDev));
}

// trim from start (in place)
static inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
    return !std::isspace(ch);
  }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
    return !std::isspace(ch);
  }).base(), s.end());
}

static std::vector<std::string> tokenizer(std::string str) {
  std::string delim = ",";
  std::vector<std::string> tokens;

  while (auto pos = str.find(",")) {
    std::string newstr = str.substr(0, pos);
    ltrim(newstr);
    rtrim(newstr);
    // Skip empty string
    if(!newstr.empty()) {
      if(std::find(tokens.begin(), tokens.end(), newstr) != tokens.end()) {
        CVAR_WARN("Duplicate token %s found in the value of %s", newstr.c_str(), str.c_str());
      }
      tokens.push_back(newstr);
    }
    str.erase(0, pos + delim.length());
    if (pos == std::string::npos) {
      break;
    }
  }
  return tokens;
}

static bool env2bool(const char *str_, const char *def) {
  std::string str(getenv(str_) ? getenv(str_) : def);
  std::transform(str.cbegin(), str.cend(), str.begin(), [](unsigned char c) { return std::tolower(c); });
  if (str == "y") return true;
  else if (str == "n") return false;
  else if (str == "yes") return true;
  else if (str == "no") return false;
  else if (str == "t") return true;
  else if (str == "f") return false;
  else if (str == "true") return true;
  else if (str == "false") return false;
  else if (str == "1") return true;
  else if (str == "0") return false;
  else CVAR_WARN_UNKNOWN_VALUE(str_, str.c_str());
  return true;
}

template <typename T>
static T env2num(const char *str, const char *def) {
  std::string in(getenv(str) ? getenv(str) : def);
  std::stringstream sstream(in);
  T ret;
  sstream >> ret;

  return ret;
}

static std::string env2str(const char *str, const char *def_) {
  const char *def = def_ ? def_ : "";
  std::string str_s = getenv(str) ? std::string(getenv(str)) : std::string(def);
  ltrim(str_s);
  rtrim(str_s);
  return str_s;
}

static std::vector<std::string> env2strlist(const char* str, const char* def_) {
  const char* def = def_ ? def_ : "";
  std::string str_s(getenv(str) ? getenv(str) : def);
  return tokenizer(str_s);
}

static std::tuple<std::string, std::vector<std::string>> env2prefixedStrlist(
    const char* str,
    const char* def_,
    std::vector<std::string> prefixes) {
  const char* def = def_ ? def_ : "";
  std::string str_s(getenv(str) ? getenv(str) : def);

  // search if any prefix is specified
  for (auto prefix : prefixes) {
    if (!str_s.compare(0, prefix.size(), prefix)) {
      // if prefix is found, convert the remaining string to stringList
      std::string slist_s = str_s.substr(prefix.size());
      return std::make_tuple(prefix, tokenizer(slist_s));
    }
  }
  // if no prefix is found, convert entire string to stringList
  return std::make_tuple("", tokenizer(str_s));
}

void initEnvSet(std::unordered_set<std::string>& env);
void readCvarEnv();

#endif /* NCCL_CVARS_BASE_H_INCLUDED */
