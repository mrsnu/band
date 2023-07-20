namespace band {
namespace tool {

#include <memory>

#include "absl/status/status.h"
#include "json.h"

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