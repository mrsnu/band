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

#include "band/device/cpu.h"

#include <cstring>
#include <mutex>  // call_once

#include "band/device/util.h"
#include "band/logger.h"

#if BAND_SUPPORT_DEVICE
#include <errno.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "cpu.h"

#endif

namespace band {
using namespace device;

#if BAND_SUPPORT_DEVICE
CpuSet::CpuSet() { DisableAll(); }

void CpuSet::Enable(int cpu) { CPU_SET(cpu, &cpu_set_); }

void CpuSet::Disable(int cpu) { CPU_CLR(cpu, &cpu_set_); }

void CpuSet::DisableAll() { CPU_ZERO(&cpu_set_); }

bool CpuSet::IsEnabled(int cpu) const { return CPU_ISSET(cpu, &cpu_set_); }

const unsigned long* CpuSet::GetMaskBits() const { return cpu_set_.__bits; }

std::vector<unsigned long> CpuSet::GetMaskBitsVector() const {
  return std::vector<unsigned long>(
      GetMaskBits(),
      GetMaskBits() + sizeof(cpu_set_.__bits) / sizeof(*cpu_set_.__bits));
}

bool CpuSet::operator==(const CpuSet& rhs) const {
  return CPU_EQUAL(&cpu_set_, &rhs.cpu_set_) != 0;
}

size_t CpuSet::NumEnabled() const {
  size_t num_enabled = 0;
  for (int i = 0; i < (int)sizeof(cpu_set_t) * 8; i++) {
    if (IsEnabled(i)) num_enabled++;
  }

  return num_enabled;
}

#else   // defined BAND_SUPPORT_DEVICE
CpuSet::CpuSet() {}

void CpuSet::Enable(int /* cpu */) {}

void CpuSet::Disable(int /* cpu */) {}

void CpuSet::DisableAll() {}

const unsigned long* CpuSet::GetMaskBits() const { return nullptr; }

std::vector<unsigned long> CpuSet::GetMaskBitsVector() const { return {}; }

bool CpuSet::operator==(const CpuSet& rhs) const { return true; }

bool CpuSet::IsEnabled(int /* cpu */) const { return true; }

size_t CpuSet::NumEnabled() const { return GetCPUCount(); }
#endif  // defined BAND_SUPPORT_DEVICE

std::string CpuSet::ToString() const {
  std::string str;
  for (int i = 0; i < GetCPUCount(); i++) {
    if (IsEnabled(i)) {
      str += std::to_string(i);
      str += ",";
    }
  }

  return str;
}

CPUMaskFlag CpuSet::GetCPUMaskFlag() const {
  for (size_t i = 0; i < EnumLength<CPUMaskFlag>(); i++) {
    const CPUMaskFlag flag = static_cast<CPUMaskFlag>(i);
    if (BandCPUMaskGetSet(flag) == *this) {
      return flag;
    }
  }
  return CPUMaskFlag::kAll;
}

static CpuSet g_thread_affinity_mask_all;
static CpuSet g_thread_affinity_mask_little;
static CpuSet g_thread_affinity_mask_big;
static CpuSet g_thread_affinity_mask_primary;
static size_t g_cpucount = GetCPUCount();

size_t GetCPUCount() {
  size_t count = 0;
#ifdef __EMSCRIPTEN__
  if (emscripten_has_threading_support())
    count = emscripten_num_logical_cores();
  else
    count = 1;
#elif defined BAND_SUPPORT_DEVICE
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

size_t GetLittleCPUCount() {
  return BandCPUMaskGetSet(CPUMaskFlag::kLittle).NumEnabled();
}

size_t GetBigCPUCount() {
  return BandCPUMaskGetSet(CPUMaskFlag::kBig).NumEnabled();
}

#if BAND_SUPPORT_DEVICE
int get_max_freq_khz(int cpuid) {
  // first try, for all possible cpu
  char path[256];
  sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state",
          cpuid);

  FILE* fp = fopen(path, "rb");

  if (!fp) {
    // second try, for online cpu
    sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",
            cpuid);
    fp = fopen(path, "rb");

    if (fp) {
      int max_freq_khz = 0;
      while (!feof(fp)) {
        int freq_khz = 0;
        int nscan = fscanf(fp, "%d %*d", &freq_khz);
        if (nscan != 1) break;

        if (freq_khz > max_freq_khz) max_freq_khz = freq_khz;
      }

      fclose(fp);

      if (max_freq_khz != 0) return max_freq_khz;

      fp = NULL;
    }

    if (!fp) {
      // third try, for online cpu
      sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
              cpuid);
      fp = fopen(path, "rb");

      if (!fp) return -1;

      int max_freq_khz = -1;
      int nscan = fscanf(fp, "%d", &max_freq_khz);
      fclose(fp);

      return max_freq_khz;
    }
  }

  int max_freq_khz = 0;
  while (!feof(fp)) {
    int freq_khz = 0;
    int nscan = fscanf(fp, "%d %*d", &freq_khz);
    if (nscan != 1) break;

    if (freq_khz > max_freq_khz) max_freq_khz = freq_khz;
  }

  fclose(fp);

  return max_freq_khz;
}

int SetSchedAffinity(const CpuSet& thread_affinity_mask) {
  // set affinity for thread
#if defined(__GLIBC__) || defined(__OHOS__)
  pid_t pid = syscall(SYS_gettid);
#else
#ifdef PI3
  pid_t pid = getpid();
#else
  pid_t pid = gettid();
#endif
#endif
  int syscallret = sched_setaffinity(pid, sizeof(cpu_set_t),
                                     &thread_affinity_mask.GetCpuSet());
  int err = errno;
  if (syscallret != 0) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR, "Set sched affinity error: %s",
                      strerror(err));
    return -1;
  }
  return 0;
}

int GetSchedAffinity(CpuSet& thread_affinity_mask) {
  // set affinity for thread
#if defined(__GLIBC__) || defined(__OHOS__)
  pid_t pid = syscall(SYS_gettid);
#else
#ifdef PI3
  pid_t pid = getpid();
#else
  pid_t pid = gettid();
#endif
#endif
  int syscallret = sched_getaffinity(pid, sizeof(cpu_set_t),
                                     &thread_affinity_mask.GetCpuSet());
  int err = errno;
  if (syscallret != 0) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR, "Get sched affinity error :%s",
                      strerror(err));
    return -1;
  }

  return 0;
}
#endif  // defined BAND_SUPPORT_DEVICE

absl::Status SetCPUThreadAffinity(const CpuSet& thread_affinity_mask) {
#if BAND_SUPPORT_DEVICE
  int ssaret = SetSchedAffinity(thread_affinity_mask);
  if (ssaret != 0) {
    return absl::InternalError("Failed to set the CPU affinity: " +
                               thread_affinity_mask.ToString());
  }
  return absl::OkStatus();
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::Status GetCPUThreadAffinity(CpuSet& thread_affinity_mask) {
#if BAND_SUPPORT_DEVICE
  int gsaret = GetSchedAffinity(thread_affinity_mask);
  std::cout << "GetCPUThreadAffinity" << thread_affinity_mask.ToString()
            << std::endl;
  if (gsaret != 0) {
    return absl::InternalError("Failed to get the CPU affinity.");
  }
  return absl::OkStatus();
#else
  return absl::UnavailableError("Device not supported");
#endif
}

int SetupThreadAffinityMasks() {
  g_thread_affinity_mask_all.DisableAll();

#if BAND_SUPPORT_DEVICE
  int max_freq_khz_min = INT_MAX;
  int max_freq_khz_max = 0;
  std::vector<int> cpu_max_freq_khz(g_cpucount);
  for (int i = 0; i < g_cpucount; i++) {
    g_thread_affinity_mask_all.Enable(i);
    int max_freq_khz = get_max_freq_khz(i);

    cpu_max_freq_khz[i] = max_freq_khz;

    if (max_freq_khz > max_freq_khz_max) max_freq_khz_max = max_freq_khz;
    if (max_freq_khz < max_freq_khz_min) max_freq_khz_min = max_freq_khz;
  }

  int max_freq_khz_medium = (max_freq_khz_min + max_freq_khz_max) / 2;
  if (max_freq_khz_medium == max_freq_khz_max) {
    g_thread_affinity_mask_little.DisableAll();
    g_thread_affinity_mask_big = g_thread_affinity_mask_all;
    return 0;
  }

  for (int i = 0; i < g_cpucount; i++) {
    if (cpu_max_freq_khz[i] < max_freq_khz_medium) {
      g_thread_affinity_mask_little.Enable(i);
    } else if (cpu_max_freq_khz[i] == max_freq_khz_max) {
      g_thread_affinity_mask_primary.Enable(i);
    } else {
      g_thread_affinity_mask_big.Enable(i);
    }
  }

  // Categorize into LITTLE and big if there is no primary core.
  if (g_thread_affinity_mask_big.NumEnabled() == 0) {
    g_thread_affinity_mask_big = g_thread_affinity_mask_primary;
    g_thread_affinity_mask_primary.DisableAll();
  }

#else
  // TODO implement me for other platforms
  g_thread_affinity_mask_little.DisableAll();
  g_thread_affinity_mask_big = g_thread_affinity_mask_all;
#endif

  return 0;
}

const CpuSet& BandCPUMaskGetSet(CPUMaskFlag flag) {
  static std::once_flag once_flag;
  std::call_once(once_flag, []() { SetupThreadAffinityMasks(); });

  switch (flag) {
    case CPUMaskFlag::kAll:
      return g_thread_affinity_mask_all;
    case CPUMaskFlag::kLittle:
      return g_thread_affinity_mask_little;
    case CPUMaskFlag::kBig:
      return g_thread_affinity_mask_big;
    case CPUMaskFlag::kPrimary:
      return g_thread_affinity_mask_primary;
    default:
      // fallback to all cores anyway
      return g_thread_affinity_mask_all;
  }
}

namespace cpu {
absl::StatusOr<size_t> GetTargetMaxFrequencyKhz(int cpu) {
#if BAND_SUPPORT_DEVICE
  if (!IsRooted()) {
    return absl::UnavailableError("Device not rooted");
  }
  return TryReadSizeT({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                           "/cpufreq/scaling_max_freq",
                       "/sys/devices/system/cpu/cpufreq/policy" +
                           std::to_string(cpu) + "/scaling_max_freq"});
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetTargetMaxFrequencyKhz(const CpuSet& cpu_set) {
#if BAND_SUPPORT_DEVICE
  if (cpu_set.NumEnabled() > 0) {
    int accumulated_frequency = 0;

    for (int i = 0; i < GetCPUCount(); i++) {
      if (cpu_set.IsEnabled(i)) {
        auto freq = GetTargetMaxFrequencyKhz(i);
        if (freq.ok()) {
          accumulated_frequency += freq.value();
        } else {
          return freq.status();
        }
      }
    }

    return accumulated_frequency / cpu_set.NumEnabled();
  } else {
    return 0;
  }
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetTargetMinFrequencyKhz(int cpu) {
#if BAND_SUPPORT_DEVICE
  return TryReadSizeT({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                           "/cpufreq/scaling_min_freq",
                       "/sys/devices/system/cpu/cpufreq/policy" +
                           std::to_string(cpu) + "/scaling_min_freq"});
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetTargetMinFrequencyKhz(const CpuSet& cpu_set) {
#if BAND_SUPPORT_DEVICE
  if (cpu_set.NumEnabled() > 0) {
    int accumulated_frequency = 0;

    for (int i = 0; i < GetCPUCount(); i++) {
      if (cpu_set.IsEnabled(i)) {
        auto freq = GetTargetMinFrequencyKhz(i);
        if (freq.ok()) {
          accumulated_frequency += freq.value();
        } else {
          return freq.status();
        }
      }
    }

    return accumulated_frequency / cpu_set.NumEnabled();
  } else {
    return 0;
  }
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetTargetFrequencyKhz(int cpu) {
#if BAND_SUPPORT_DEVICE
  return TryReadSizeT({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                           "/cpufreq/scaling_cur_freq",
                       "/sys/devices/system/cpu/cpufreq/policy" +
                           std::to_string(cpu) + "/scaling_cur_freq"});
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetTargetFrequencyKhz(const CpuSet& cpu_set) {
#if BAND_SUPPORT_DEVICE
  if (cpu_set.NumEnabled() > 0) {
    int accumulated_frequency = 0;

    for (int i = 0; i < GetCPUCount(); i++) {
      if (cpu_set.IsEnabled(i)) {
        auto freq = GetTargetFrequencyKhz(i);
        if (freq.ok()) {
          accumulated_frequency += freq.value();
        } else {
          return freq.status();
        }
      }
    }

    return accumulated_frequency / cpu_set.NumEnabled();
  } else {
    return 0;
  }
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetFrequencyKhz(int cpu) {
#if BAND_SUPPORT_DEVICE
  if (!IsRooted()) {
    return absl::UnavailableError("Device not rooted");
  }
  return TryReadSizeT({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                           "/cpufreq/cpuinfo_cur_freq",
                       "/sys/devices/system/cpu/cpufreq/policy" +
                           std::to_string(cpu) + "/cpuinfo_cur_freq"});
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetFrequencyKhz(const CpuSet& cpu_set) {
#if BAND_SUPPORT_DEVICE
  if (!IsRooted()) {
    return absl::UnavailableError("Device not rooted");
  }

  if (cpu_set.NumEnabled() > 0) {
    int accumulated_frequency = 0;

    for (int i = 0; i < GetCPUCount(); i++) {
      if (cpu_set.IsEnabled(i)) {
        auto freq = GetFrequencyKhz(i);
        if (freq.ok()) {
          accumulated_frequency += freq.value();
        } else {
          return freq.status();
        }
      }
    }

    return accumulated_frequency / cpu_set.NumEnabled();
  } else {
    return 0;
  }
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<std::vector<size_t> > GetAvailableFrequenciesKhz(
    const CpuSet& cpu_set) {
#if BAND_SUPPORT_DEVICE
  for (int cpu = 0; cpu < GetCPUCount(); cpu++) {
    // Assuming that there is one cluster group
    if (cpu_set.IsEnabled(cpu)) {
      return TryReadSizeTs(
          {"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
               "/cpufreq/scaling_available_frequencies",
           "/sys/devices/system/cpu/cpufreq/policy" + std::to_string(cpu) +
               "/scaling_available_frequencies"});
    }
  }
  return absl::NotFoundError("No available frequencies found");
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetUpTransitionLatencyMs(int cpu) {
#if BAND_SUPPORT_DEVICE
  absl::StatusOr<size_t> cpu_transition =
      TryReadSizeT({"/sys/devices/system/cpu/cpufreq/policy" +
                    std::to_string(cpu) + "/schedutil/up_rate_limit_us"});
  if (cpu_transition.ok()) {
    return cpu_transition.value() == 0 ? 0 : cpu_transition.value() / 1000;
  }

  cpu_transition =
      TryReadSizeT({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                    "/cpufreq/cpuinfo_transition_latency"});
  if (cpu_transition.ok()) {
    return cpu_transition.value() == 0 ? 0 : cpu_transition.value() / 1000000;
  }

  return absl::NotFoundError("No up transition latency found");
#else
  return absl::UnavailableError("Device not supported");
#endif
}

// Assuming that there is one cluster group
absl::StatusOr<size_t> GetUpTransitionLatencyMs(const CpuSet& cpu_set) {
#if BAND_SUPPORT_DEVICE
  if (cpu_set.NumEnabled() > 0) {
    for (int i = 0; i < GetCPUCount(); i++) {
      if (cpu_set.IsEnabled(i)) {
        return GetUpTransitionLatencyMs(i);
      }
    }
  } else {
    return 0;
  }
#endif
  return absl::UnavailableError("Device not supported");
}

absl::StatusOr<size_t> GetDownTransitionLatencyMs(int cpu) {
#if BAND_SUPPORT_DEVICE
  absl::StatusOr<size_t> cpu_transition =
      TryReadSizeT({"/sys/devices/system/cpu/cpufreq/policy" +
                    std::to_string(cpu) + "/schedutil/down_rate_limit_us"});
  if (cpu_transition.ok()) {
    return cpu_transition.value() == 0 ? 0 : cpu_transition.value() / 1000;
  }

  cpu_transition =
      TryReadSizeT({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                    "/cpufreq/cpuinfo_transition_latency"});

  if (cpu_transition.ok()) {
    return cpu_transition.value() == 0 ? 0 : cpu_transition.value() / 1000000;
  }

  return absl::NotFoundError("No down transition latency found");
#else
  return absl::UnavailableError("Device not supported");
#endif
}

// Assuming that there is one cluster group
absl::StatusOr<size_t> GetDownTransitionLatencyMs(const CpuSet& cpu_set) {
#if BAND_SUPPORT_DEVICE
  for (int i = 0; i < GetCPUCount(); i++) {
    if (cpu_set.IsEnabled(i)) {
      return GetDownTransitionLatencyMs(i);
    }
  }
  return 0;
#else
  return absl::UnavailableError("Device not supported");
#endif
}

// Total transition count
// Note that cores in same cluster (little/big/primary)
// shares this value
absl::StatusOr<size_t> GetTotalTransitionCount(int cpu) {
#if BAND_SUPPORT_DEVICE
  return TryReadSizeT({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                           "/cpufreq/stats/total_trans",
                       "/sys/devices/system/cpu/cpufreq/policy" +
                           std::to_string(cpu) + "/stats/total_trans"});
#else
  return absl::UnavailableError("Device not supported");
#endif
}

absl::StatusOr<size_t> GetTotalTransitionCount(const CpuSet& cpu_set) {
#if BAND_SUPPORT_DEVICE
  if (cpu_set.NumEnabled() > 0) {
    int accumulated_transition_count = 0;

    for (int i = 0; i < GetCPUCount(); i++) {
      if (cpu_set.IsEnabled(i)) {
        absl::StatusOr<size_t> transition_count = GetTotalTransitionCount(i);
        if (!transition_count.ok()) {
          return transition_count.status();
        } else {
          accumulated_transition_count += transition_count.value();
        }
      }
    }

    return accumulated_transition_count / cpu_set.NumEnabled();
  } else {
    return 0;
  }
#else
  return absl::UnavailableError("Device not supported");
#endif
}
}  // namespace cpu
}  // namespace band
