#include "band/tool/runner.h"

#include "band/common.h"
#include "runner.h"

namespace band {
namespace tool {

IRunner::~IRunner() {
  for (size_t i = 0; i < children_.size(); i++) {
    delete children_[i];
  }
}

absl::Status IRunner::Initialize(const Json::Value& root) {
  return absl::OkStatus();
}

void IRunner::Join() {
  for (size_t i = 0; i < children_.size(); i++) {
    children_[i]->Join();
  }
}
absl::Status IRunner::LogResults(size_t instance_id) {
  for (size_t i = 0; i < children_.size(); i++) {
    RETURN_IF_ERROR(children_[i]->LogResults(i));
  }
  return absl::OkStatus();
}

}  // namespace tool
}  // namespace band