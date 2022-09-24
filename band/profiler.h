#ifndef BAND_PROFILER_H_
#define BAND_PROFILER_H_

#include <map>

#include "band/common.h"
#include "band/config.h"

namespace Band {
class Profiler {
public:
  Profiler() = default;
  BandStatus Init(const ProfileConfig &config);
  void UpdateLatency(const SubgraphKey &key, int64_t latency);

  int64_t GetProfiled(const SubgraphKey &key) const;
  int64_t GetExpected(const SubgraphKey &key) const;

private:
  // latency in microseconds
  struct Latency {
    int64_t profiled;
    int64_t moving_averaged;
  };

  // Path to the profile data.
  // The data in the path will be read during initial phase, and also
  // will be updated at the end of the run.
  std::string profile_data_path_;

  // The contents of the file at `profile_data_path_`.
  // We keep this separately from `profile_database_`, since we cannot
  // immediately put `profile_data_path_`'s contents into `profile_database_`
  // because the model name --> int mapping is not available at init time.
  Json::Value profile_database_json_;

  std::map<SubgraphKey, Latency> profile_database_;
  float profile_smoothing_factor_;

  bool profile_online_;
  int profile_num_warmups_;
  int profile_num_runs_;
  std::vector<int> profile_copy_computation_ratio_;
};
} // namespace Band

#endif // BAND_PROFILER_H_