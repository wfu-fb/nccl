/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "param.h"
#include "debug.h"
#include "nccl_cvars.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <pwd.h>

const char* userHomeDir() {
  struct passwd *pwUser = getpwuid(getuid());
  return pwUser == NULL ? NULL : pwUser->pw_dir;
}

void setEnvFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char *line = NULL;
  char envVar[1024];
  char envValue[1024];
  size_t n = 0;
  ssize_t read;
  while ((read = getline(&line, &n, file)) != -1) {
    if (line[read-1] == '\n') line[read-1] = '\0';
    int s=0; // Env Var Size
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1023,s));
    envVar[s] = '\0';
    s++;
    strncpy(envValue, line+s, 1023);
    envValue[1023]='\0';
    setenv(envVar, envValue, 0);
    //printf("%s : %s->%s\n", fileName, envVar, envValue);
  }
  if (line) free(line);
  fclose(file);
}

void initEnv() {
  char confFilePath[1024];
  const char * userDir = userHomeDir();
  if (userDir) {
    sprintf(confFilePath, "%s/.nccl.conf", userDir);
    setEnvFile(confFilePath);
  }
  sprintf(confFilePath, "/etc/nccl.conf");
  setEnvFile(confFilePath);
  ncclCvarInit();
}

void initEnvOnce() {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, initEnv);
}
