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

#if BAND_IS_MOBILE
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

#if BAND_IS_MOBILE
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
}  // namespace band

#endif  // BAND_CPU_H_
