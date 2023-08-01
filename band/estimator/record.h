#ifndef BAND_ESTIMATOR_RECORD_H_
#define BAND_ESTIMATOR_RECORD_H_

namespace band {

struct LatencyRecord {
  int64_t profiled = 0;
  int64_t expected = 0;
};
  
}  // namespace band

#endif  // BAND_ESTIMATOR_RECORD_H_