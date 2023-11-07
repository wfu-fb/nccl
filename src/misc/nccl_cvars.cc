// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Automatically generated
//   by ./maint/extractcvars.py
// DO NOT EDIT!!!

#include <string>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <set>
#include <cstring>
#include "nccl_cvars.h"
#include "debug.h"

static std::set<std::string> tokenizer(const char *str_, const char *def_) {
  const char *def = def_ ? def_ : "";
  std::string str(getenv(str_) ? getenv(str_) : def);
  std::string delim = ",";
  std::set<std::string> tokens;

  while (auto pos = str.find(",")) {
    std::string newstr = str.substr(0, pos);
    if (tokens.find(newstr) != tokens.end()) {
      WARN("Duplicate token %s found in the value of %s", newstr.c_str(), str_);
    }
    tokens.insert(newstr);
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
  else WARN("Unrecognized value for env %s\n", str_);
  return true;
}

static int env2int(const char *str, const char *def) {
  return getenv(str) ? atoi(getenv(str)) : atoi(def);
}

static std::string env2str(const char *str, const char *def_) {
  const char *def = def_ ? def_ : "";
  return getenv(str) ? std::string(getenv(str)) : std::string(def);
}

static std::set<std::string> env2strlist(const char *str, const char *def) {
  return tokenizer(str, def);
}

bool NCCL_CVAR_DDA_ALLREDUCE_LARGE_MESSAGE_HCM;

int NCCL_CVAR_DDA_ALLREDUCE_TMPBUFF_SIZE;

int NCCL_CVAR_DDA_MAX_RANKS;

enum NCCL_CVAR_ALLREDUCE_ALGO NCCL_CVAR_ALLREDUCE_ALGO;

int NCCL_CVAR_DDA_ALLREDUCE_MAX_BLOCKS;

int NCCL_CVAR_DDA_ALLREDUCE_TREE_THRESHOLD_NVS;

int NCCL_CVAR_DDA_ALLREDUCE_TREE_THRESHOLD_HCM;

bool NCCL_CVAR_DDA_FORCE_P2P_ACCESS;

extern char **environ;

void ncclCvarInit() {
  std::unordered_set<std::string> env;
  env.insert("NCCL_CVAR_DDA_ALLREDUCE_LARGE_MESSAGE_HCM");
  env.insert("NCCL_CVAR_DDA_ALLREDUCE_TMPBUFF_SIZE");
  env.insert("NCCL_CVAR_DDA_MAX_RANKS");
  env.insert("NCCL_CVAR_ALLREDUCE_ALGO");
  env.insert("NCCL_CVAR_DDA_ALLREDUCE_MAX_BLOCKS");
  env.insert("NCCL_CVAR_DDA_ALLREDUCE_TREE_THRESHOLD_NVS");
  env.insert("NCCL_CVAR_DDA_ALLREDUCE_TREE_THRESHOLD_HCM");
  env.insert("NCCL_CVAR_DDA_FORCE_P2P_ACCESS");
  env.insert("NCCL_ALGO");
  env.insert("NCCL_COLLNET_ENABLE");
  env.insert("NCCL_COLLTRACE_LOCAL_SUBDIR");
  env.insert("NCCL_COMM_ID");
  env.insert("NCCL_CUDA_PATH");
  env.insert("NCCL_DEBUG");
  env.insert("NCCL_DEBUG_FILE");
  env.insert("NCCL_DEBUG_SUBSYS");
  env.insert("NCCL_GRAPH_DUMP_FILE");
  env.insert("NCCL_GRAPH_FILE");
  env.insert("NCCL_HOSTID");
  env.insert("NCCL_IB_GID_INDEX");
  env.insert("NCCL_IB_TC");
  env.insert("NCCL_LAUNCH_MODE");
  env.insert("NCCL_NET");
  env.insert("NCCL_NET_PLUGIN");
  env.insert("NCCL_NSOCKS_PERTHREAD");
  env.insert("NCCL_PROTO");
  env.insert("NCCL_PROXY_PROFILE");
  env.insert("NCCL_SHM_DISABLE");
  env.insert("NCCL_SOCKET_FAMILY");
  env.insert("NCCL_SOCKET_IFNAME");
  env.insert("NCCL_SOCKET_NTHREADS");
  env.insert("NCCL_THREAD_THRESHOLDS");
  env.insert("NCCL_TOPO_DUMP_FILE");
  env.insert("NCCL_TOPO_FILE");
  env.insert("NCCL_TUNER_PLUGIN");

  char **s = environ;
  for (; *s; s++) {
    if (!strncmp(*s, "NCCL_", strlen("NCCL_"))) {
      std::string str(*s);
      str = str.substr(0, str.find("="));
      if (env.find(str) == env.end()) {
        WARN("Unknown env %s in the NCCL namespace\n", str.c_str());
      }
    }
  }

  NCCL_CVAR_DDA_ALLREDUCE_LARGE_MESSAGE_HCM = env2bool("NCCL_CVAR_DDA_ALLREDUCE_LARGE_MESSAGE_HCM", "False");

  NCCL_CVAR_DDA_ALLREDUCE_TMPBUFF_SIZE = env2int("NCCL_CVAR_DDA_ALLREDUCE_TMPBUFF_SIZE", "33554432");

  NCCL_CVAR_DDA_MAX_RANKS = env2int("NCCL_CVAR_DDA_MAX_RANKS", "16");

  if (getenv("NCCL_CVAR_ALLREDUCE_ALGO") == nullptr) {
    NCCL_CVAR_ALLREDUCE_ALGO = NCCL_CVAR_ALLREDUCE_ALGO::orig;
  } else {
    std::string str(getenv("NCCL_CVAR_ALLREDUCE_ALGO"));
    if (str == std::string("orig")) {
      NCCL_CVAR_ALLREDUCE_ALGO = NCCL_CVAR_ALLREDUCE_ALGO::orig;
    } else if (str == std::string("dda")) {
      NCCL_CVAR_ALLREDUCE_ALGO = NCCL_CVAR_ALLREDUCE_ALGO::dda;
    } else {
      WARN("Unknown value %s for env NCCL_CVAR_ALLREDUCE_ALGO", str.c_str());
    }
  }

  NCCL_CVAR_DDA_ALLREDUCE_MAX_BLOCKS = env2int("NCCL_CVAR_DDA_ALLREDUCE_MAX_BLOCKS", "1");

  NCCL_CVAR_DDA_ALLREDUCE_TREE_THRESHOLD_NVS = env2int("NCCL_CVAR_DDA_ALLREDUCE_TREE_THRESHOLD_NVS", "262144");

  NCCL_CVAR_DDA_ALLREDUCE_TREE_THRESHOLD_HCM = env2int("NCCL_CVAR_DDA_ALLREDUCE_TREE_THRESHOLD_HCM", "65536");

  NCCL_CVAR_DDA_FORCE_P2P_ACCESS = env2bool("NCCL_CVAR_DDA_FORCE_P2P_ACCESS", "False");

}
