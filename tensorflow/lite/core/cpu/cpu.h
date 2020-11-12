#ifndef TENSORFLOW_LITE_CORE_CPU_CPU_H_
#define TENSORFLOW_LITE_CORE_CPU_CPU_H_

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <limits.h>

#if defined __ANDROID__ || defined __linux__
#include <sched.h> // cpu_set_t
#endif

namespace tflite {
namespace impl {

class CpuSet {
public:
    CpuSet();
    void enable(int cpu);
    void disable(int cpu);
    void disable_all();
    bool is_enabled(int cpu) const;
    int num_enabled() const;

public:
#if defined __ANDROID__ || defined __linux__
    cpu_set_t cpu_set;
#endif
};

// cpu info
int get_cpu_count();
int get_little_cpu_count();
int get_big_cpu_count();

// convenient wrapper
const CpuSet& get_cpu_thread_affinity_mask(int powersave);

// set explicit thread affinity
int set_cpu_thread_affinity(const CpuSet& thread_affinity_mask);

} // namespace impl
} // nmaespace tflite

#endif // TENSORFLOW_LITE_CPU_CPU_H_
