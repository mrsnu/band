#ifndef BAND_TOOL_RUNNER_H_
#define BAND_TOOL_RUNNER_H_

#include "band/json_util.h"

namespace band {
namespace tool {

class IRunner {
 public:
  virtual ~IRunner();
  virtual absl::Status Initialize(const Json::Value& root);
  virtual absl::Status Run() = 0;
  virtual void Join();
  virtual absl::Status LogResults(size_t instance_id);

 protected:
  std::vector<IRunner*> children_;
};

}  // namespace tool
}  // namespace band

#endif  // BAND_TOOL_RUNNER_H_