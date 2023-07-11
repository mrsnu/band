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
#include "absl/status/statusor.h"
#include "band/common.h"
#include "band/device/util.h"

#if BAND_SUPPORT_DEVICE
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
  size_t NumEnabled() const;
  CPUMaskFlag GetCPUMaskFlag() const;
  const unsigned long* GetMaskBits() const;
  std::vector<unsigned long> GetMaskBitsVector() const;
  std::string ToString() const;
  bool operator==(const CpuSet& rhs) const;

#if BAND_SUPPORT_DEVICE
  const cpu_set_t& GetCpuSet() const { return cpu_set_; }
  cpu_set_t& GetCpuSet() { return cpu_set_; }

 private:
  cpu_set_t cpu_set_;
#endif
};

// cpu info
size_t GetCPUCount();
size_t GetLittleCPUCount();
size_t GetBigCPUCount();

// set explicit thread affinity
absl::Status SetCPUThreadAffinity(const CpuSet& thread_affinity_mask);
absl::Status GetCPUThreadAffinity(CpuSet& thread_affinity_mask);

// convenient wrapper
const CpuSet& BandCPUMaskGetSet(CPUMaskFlag flag);

// Wrap frequency-related helper functions for consistency with *pu
namespace cpu {

// Get scaling frequency (current target frequency of the governor)
absl::StatusOr<size_t> GetTargetFrequencyKhz(int cpu);
absl::StatusOr<size_t> GetTargetFrequencyKhz(const CpuSet& cpu_set);

// Get scaling max frequency (current target frequency of the governor)
// This requires sudo in some devices (e.g., Pixel 6)
absl::StatusOr<size_t> GetTargetMaxFrequencyKhz(int cpu);
absl::StatusOr<size_t> GetTargetMaxFrequencyKhz(const CpuSet& cpu_set);

// Get scaling min frequency (current target frequency of the governor)
absl::StatusOr<size_t> GetTargetMinFrequencyKhz(int cpu);
absl::StatusOr<size_t> GetTargetMinFrequencyKhz(const CpuSet& cpu_set);

// Get current frequency (requires sudo)
absl::StatusOr<size_t> GetFrequencyKhz(int cpu);
absl::StatusOr<size_t> GetFrequencyKhz(const CpuSet& cpu_set);

absl::StatusOr<std::vector<size_t>> GetAvailableFrequenciesKhz(
    const CpuSet& cpu_set);

// Time interval limit of frequency rise
absl::StatusOr<size_t> GetUpTransitionLatencyMs(int cpu);
absl::StatusOr<size_t> GetUpTransitionLatencyMs(const CpuSet& cpu_set);

// Time interval limit of frequency down
absl::StatusOr<size_t> GetDownTransitionLatencyMs(int cpu);
absl::StatusOr<size_t> GetDownTransitionLatencyMs(const CpuSet& cpu_set);

// Total transition count
// Note that cores in same cluster (little/big/primary)
// shares this value
absl::StatusOr<size_t> GetTotalTransitionCount(int cpu);
absl::StatusOr<size_t> GetTotalTransitionCount(const CpuSet& cpu_set);
}  // namespace cpu

}  // namespace band

#endif  // BAND_CPU_H_
