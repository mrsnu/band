#ifndef BAND_ESTIMATOR_ESTIMATOR_H_
#define BAND_ESTIMATOR_ESTIMATOR_H_

#include <chrono>
#include <unordered_map>

#include "band/common.h"
#include "band/config.h"

#include "absl/status/status.h"

namespace band {

class IEngine;

template<typename EstimatorKey>
class IEstimator {
 public:
  explicit IEstimator(IEngine* engine) : engine_(engine) {}
  virtual void Update(const EstimatorKey& key, int64_t new_value) = 0;
  virtual absl::Status Profile(ModelId model_id) = 0;
  virtual int64_t GetProfiled(const EstimatorKey& key) const = 0;
  virtual int64_t GetExpected(const EstimatorKey& key) const = 0;

  virtual absl::Status DumpProfile() = 0;

 protected:
  IEngine* engine_;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_ESTIMATOR_H_