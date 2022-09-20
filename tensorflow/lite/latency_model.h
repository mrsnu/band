#ifndef TENSORFLOW_LITE_LATENCY_MODEL_H_
#define TENSORFLOW_LITE_LATENCY_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"

namespace tflite {
namespace impl {

// 
class LatencyModel {
 public:
  void ClearHistory();
  void ClearHistoryAll();
  void DumpAllHistory();

 private:
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_LATENCY_MODEL_H_