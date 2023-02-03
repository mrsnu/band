// This source code is created by Tencent's NCNN project.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "ruy/cpu.h"

#if defined __ANDROID__ || defined __linux__
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace ruy {
#if defined __ANDROID__ || defined __linux__
CpuSet::CpuSet() {
  for (int i = 0; i < GetCPUCount(); i++) {
    Enable(i);
  }; 
}

CpuSet::CpuSet(const unsigned long* mask) {
  memcpy(cpu_set_.__bits, mask, sizeof(cpu_set_.__bits));
}

void CpuSet::Enable(int cpu) { CPU_SET(cpu, &cpu_set_); }

void CpuSet::Disable(int cpu) { CPU_CLR(cpu, &cpu_set_); }

void CpuSet::DisableAll() { CPU_ZERO(&cpu_set_); }

bool CpuSet::IsEnabled(int cpu) const { return CPU_ISSET(cpu, &cpu_set_); }

int CpuSet::NumEnabled() const {
  int NumEnabled = 0;
  for (int i = 0; i < (int)sizeof(cpu_set_t) * 8; i++) {
    if (IsEnabled(i)) NumEnabled++;
  }

  return NumEnabled;
}

bool CpuSet::SetAffinity() const {
  return sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set_) == 0; 
}

#else   // defined __ANDROID__ || defined __linux__
CpuSet::CpuSet() {}

CpuSet::CpuSet(const unsigned long* mask) {}

void CpuSet::Enable(int /* cpu */) {}

void CpuSet::Disable(int /* cpu */) {}

void CpuSet::DisableAll() {}

bool CpuSet::IsEnabled(int /* cpu */) const { return true; }

int CpuSet::NumEnabled() const { return GetCPUCount(); }

bool CpuSet::SetAffinity() const { return true; }
#endif  // defined __ANDROID__ || defined __linux__

int GetCPUCount() {
  int count = 0;
#ifdef __EMSCRIPTEN__
  if (emscripten_has_threading_support())
    count = emscripten_num_logical_cores();
  else
    count = 1;
#elif defined __ANDROID__ || defined __linux__
  // get cpu count from /proc/cpuinfo
  FILE* fp = fopen("/proc/cpuinfo", "rb");
  if (!fp) return 1;

  char line[1024];
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) break;

    if (memcmp(line, "processor", 9) == 0) {
      count++;
    }
  }

  fclose(fp);
#elif __IOS__
  size_t len = sizeof(count);
  sysctlbyname("hw.ncpu", &count, &len, NULL, 0);
#else
  count = 1;
#endif

  if (count < 1) count = 1;

  return count;
}
}  // namespace ruy