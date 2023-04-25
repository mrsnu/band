#ifndef BAND_TOOL_BENCHMARK_UTIL_H_
#define BAND_TOOL_BENCHMARK_UTIL_H_

namespace band {
namespace tool {

// motivated from /tensorflow/lite/tools/benchmark
template <typename T, typename Distribution>
void CreateRandomTensorData(void* target_ptr, int num_elements,
                            Distribution distribution) {
  std::mt19937 random_engine;
  T* target_head = static_cast<T*>(target_ptr);
  std::generate_n(target_head, num_elements, [&]() {
    return static_cast<T>(distribution(random_engine));
  });
}

}
}

#endif  // BAND_TOOL_BENCHMARK_UTIL_H_