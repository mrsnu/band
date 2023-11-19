#ifndef BAND_ESTIMATOR_ESTIMATOR_H_
#define BAND_ESTIMATOR_ESTIMATOR_H_

#include <chrono>
#include <unordered_map>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"

namespace band {

class IEngine;

template <typename EstimatorKey, typename EstimatorInput,
          typename EstimatorOutput>
class IEstimator {
 public:
  explicit IEstimator(IEngine* engine) : engine_(engine) {}
  virtual void Update(const SubgraphKey& key, JobId event_handle) = 0;
  virtual EstimatorOutput GetProfiled(const SubgraphKey& key) const = 0;
  virtual EstimatorOutput GetExpected(const EstimatorKey& key) const = 0;

  virtual absl::Status LoadModel(std::string profile_path) = 0;
  virtual absl::Status DumpModel(std::string path) = 0;

 protected:
  IEngine* engine_;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_ESTIMATOR_H_