// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <string>
#include <iostream>
#include <unordered_set>
#include "nccl_cvars_base.h"
#include "nccl_cvars.h"
#include "debug.h"
#include "checks.h"

extern char **environ;

void ncclCvarInit() {
  std::unordered_set<std::string> env;
  initEnvSet(env);

  initCvarLogger();

  // Check if any NCCL_ env var is not in allow list
  char **s = environ;
  for (; *s; s++) {
    if (!strncmp(*s, "NCCL_", strlen("NCCL_"))) {
      std::string str(*s);
      str = str.substr(0, str.find("="));
      if (env.find(str) == env.end()) {
        CVAR_WARN("Unknown env %s in the NCCL namespace", str.c_str());
      }
    }
  }

  readCvarEnv();
}
