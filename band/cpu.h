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
#include "band/c/common.h"

#if defined __ANDROID__ || defined __linux__
#include <sched.h>  // cpu_set_t
#endif

namespace Band {

typedef enum {
  kBandAll = 0,
  kBandLittle = 1,
  kBandBig = 2,
  kBandPrimary = 3,
  kBandNumCpuMasks = 4
} BandCPUMaskFlags;

class CpuSet {
 public:
  CpuSet();
  void Enable(int cpu);
  void Disable(int cpu);
  void DisableAll();
  bool IsEnabled(int cpu) const;
  int NumEnabled() const;
  const unsigned long* GetMaskBits() const;
#if defined __ANDROID__ || defined __linux__
  const cpu_set_t& GetCpuSet() const { return cpu_set_; }

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
const CpuSet& BandCPUMaskGetSet(BandCPUMaskFlags flag);
const char* BandCPUMaskGetName(BandCPUMaskFlags flag);
const BandCPUMaskFlags BandCPUMaskGetMask(const char* name);

}  // namespace Band

#endif  // BAND_CPU_H_
