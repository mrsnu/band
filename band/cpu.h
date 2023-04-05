// This source code is created by Tencent's NCNN project.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#ifndef BAND_CPU_H_
#define BAND_CPU_H_

#include <limits.h>
#include <stddef.h>
#include <stdio.h>

#include <vector>

#include "absl/status/status.h"

#include "band/common.h"

#ifdef __ANDROID__
#define _BAND_SUPPORT_THREAD_AFFINITY
#endif

#ifdef _BAND_SUPPORT_THREAD_AFFINITY
#include <sched.h>  // cpu_set_t
#endif

namespace band {

class CpuSet {
 public:
  CpuSet();
  void Enable(int cpu);
  void Disable(int cpu);
  void DisableAll();
  bool IsEnabled(int cpu) const;
  int NumEnabled() const;
  CPUMaskFlags GetCPUMaskFlag() const;
  const unsigned long* GetMaskBits() const;
  bool operator==(const CpuSet& rhs) const;

#if defined _BAND_SUPPORT_THREAD_AFFINITY
  const cpu_set_t& GetCpuSet() const { return cpu_set_; }
  cpu_set_t& GetCpuSet() { return cpu_set_; }

 private:
  cpu_set_t cpu_set_;
#endif
};

// cpu info
int GetCPUCount();
int GetLittleCPUCount();
int GetBigCPUCount();

// set explicit thread affinity
absl::Status SetCPUThreadAffinity(const CpuSet& thread_affinity_mask);
absl::Status GetCPUThreadAffinity(CpuSet& thread_affinity_mask);

// convenient wrapper
const CpuSet& BandCPUMaskGetSet(CPUMaskFlags flag);
const char* BandCPUMaskGetName(CPUMaskFlags flag);
const CPUMaskFlags BandCPUMaskGetFlag(const char* name);

}  // namespace band

#endif  // BAND_CPU_H_
