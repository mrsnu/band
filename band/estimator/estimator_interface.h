#ifndef BAND_ESTIMATOR_ESTIMATOR_H_
#define BAND_ESTIMATOR_ESTIMATOR_H_

#include <chrono>
#include <unordered_map>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"

namespace band {

class IEngine;

template <typename EstimatorKey, typename EstimatorValue>
class IEstimator {
 public:
  explicit IEstimator(IEngine* engine) : engine_(engine) {}
  virtual void Update(const EstimatorKey& key, EstimatorValue new_value) = 0;
  virtual void UpdateWithEvent(const EstimatorKey& key, size_t event_handle) = 0;
  virtual EstimatorValue GetProfiled(const EstimatorKey& key) const = 0;
  virtual EstimatorValue GetExpected(const EstimatorKey& key) const = 0;

  virtual absl::Status LoadProfile(std::string profile_path) = 0;
  virtual absl::Status DumpProfile(std::string path) = 0;

 protected:
  IEngine* engine_;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_ESTIMATOR_H_