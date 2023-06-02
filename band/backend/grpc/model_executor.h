#ifndef BAND_BACKEND_GRPC_MODEL_EXECUTOR_H_
#define BAND_BACKEND_GRPC_MODEL_EXECUTOR_H_

#include "band/interface/model_executor.h"

namespace band {
namespace grpc {

class GrpcModelExecutor : public interface::IModelExecutor {
 public:
  using interface::IModelExecutor::IModelExecutor;
  ~GrpcModelExecutor() override;

  absl::StatusOr<ModelSpec> InvestigateModelSpec(interface::IModel* model) override;
  absl::Status PrepareSubgraph(interface::IModel* model, std::set<int> ops = {},
                               std::set<int> unit_indices = {}) override;

  BackendType GetBackendType() const override;
  const std::vector<int>& GetInputs(const SubgraphKey& key) const override;
  const std::vector<int>& GetOutputs(const SubgraphKey& key) const override;
  const char* GetInputName(const SubgraphKey& key, int index) const override;
  const char* GetOutputName(const SubgraphKey& key, int index) const override;
  size_t GetNumTensors(const SubgraphKey& key) const override;
  size_t GetNumNodes(const SubgraphKey& key) const override;

  std::shared_ptr<interface::ITensorView> GetTensorView(const SubgraphKey& key,
                                                        int index) override;
  SubgraphKey GetLargestSubgraphKey() const override;
  bool HasSubgraph(const SubgraphKey& key) const override;

  absl::Status ExecuteSubgraph(const SubgraphKey& key) override;
  void ForEachSubgraph(std::function<void(const SubgraphKey&)> iterator) override;

 private:
  std::vector<int> inputs_;
  std::vector<int> outputs_;
};

}
}

#endif  // BAND_BACKEND_GRPC_MODEL_EXECUTOR_H_