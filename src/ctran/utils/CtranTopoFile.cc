#include "CtranTopoFile.h"
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <string>
#include "nccl_cvars.h"
#include "comm.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_CTRAN_TOPO_FILE
   type        : string
   default     : ""
   description : |-
     File that contains topology information in KEY=VALUE format

 - name        : NCCL_CTRAN_TOPO_FILE_KEYS
   type        : stringlist
   default     : ""
   description : |-
     Comma-separated list of keys to look for in NCCL_CTRAN_TOPO_FILE. In order,
     these will be used to determine the hierarchical configuration of the cluster.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

int64_t getNumFromDigits(const std::string& s) {
  std::string digits;
  int64_t num{-1};
  bool foundEq{false};
  for (const auto& c : s) {
    // Read until first equals sign
    if (!foundEq && c == '=') {
      foundEq = true;
    } else if (!foundEq) {
      continue;
    }

    // Convert everything after that to a number
    if (std::isdigit(c)) {
      digits += c;
    }
  }
  if (digits.size() > 0) {
    num = stoll(digits);
  }
  return num;
}

std::vector<int64_t> getTopoValsFromFile() {
  std::vector<int64_t> vals;
  std::ifstream infile;
  std::string line;
  std::vector<std::string> keys = NCCL_CTRAN_TOPO_FILE_KEYS;
  std::vector<std::string> keysWithEq;
  std::unordered_map<std::string, int64_t> keyVals;

  for (const auto& key: keys) {
    keyVals[key] = -1;
    // Want to exact match on key, so append equals sign to it
    keysWithEq.push_back(key + "=");
  }

  infile.open(NCCL_CTRAN_TOPO_FILE);
  if (!infile.is_open()) {
    return {};
  }
  while (getline(infile, line)) {
    for (int i=0; i<keys.size(); i++) {
      if (line.find(keysWithEq[i], 0) != std::string::npos) {
        auto& key = keys[i];
        keyVals[key] = getNumFromDigits(line);
        INFO(NCCL_INIT, "CTRAN topoInfo -- Key %s = %ld", key.c_str(), keyVals[key]);
      }
    }
  }

  for (const auto& key : keys) {
    vals.push_back(keyVals[key]);
  }

  return vals;
}
