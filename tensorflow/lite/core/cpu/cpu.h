#ifndef TENSORFLOW_LITE_CORE_CPU_CPU_H_
#define TENSORFLOW_LITE_CORE_CPU_CPU_H_

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <limits.h>
#include "tensorflow/lite/c/common.h"

#if defined __ANDROID__ || defined __linux__
#include <sched.h> // cpu_set_t
#endif

namespace tflite {
namespace impl {

typedef enum {
    kTfLiteAll,
    kTfLiteLittle,
    kTfLiteBig,
    kTfLitePrimary,
    kTfLiteLittle1,
    kTfLiteLittle2,
    kTfLiteLittle3,
    kTfLiteLittle4,
    kTfLiteBig1,
    kTfLiteBig2,
    kTfLiteBig3,
    kTfLiteBig4,
} TFLiteCPUMasks;

class CpuSet {
public:
    CpuSet();
    void Enable(int cpu);
    void Disable(int cpu);
    void DisableAll();
    bool IsEnabled(int cpu) const;
    int NumEnabled() const;
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
TfLiteStatus SetCPUThreadAffinity(const CpuSet& thread_affinity_mask);
int SetupThreadAffinityMasks();

// convenient wrapper
const CpuSet& GetCPUThreadAffinityMask(TFLiteCPUMasks mask);
const char* GetCPUThreadAffinityMaskString(TFLiteCPUMasks mask);

static int big_core_refcounter = 0;

} // namespace impl
} // nmaespace tflite

#endif // TENSORFLOW_LITE_CPU_CPU_H_
