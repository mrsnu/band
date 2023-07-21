#ifndef BAND_TOOL_RUNNER_H_
#define BAND_TOOL_RUNNER_H_

namespace band {
namespace tool {

#include <json/json.h>

#include <memory>

#include "absl/status/status.h"

class IRunner {
 public:
  virtual ~IRunner() = default;
  virtual absl::Status Initialize(const Json::Value& root);
  virtual absl::Status Run() = 0;
  virtual void Join();
  virtual absl::Status LogResults(size_t instance_id);

 protected:
  std::vector<std::unique_ptr<IRunner>> children_;
};

}  // namespace tool
}  // namespace band

#endif  // BAND_TOOL_RUNNER_H_