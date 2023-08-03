#ifndef BAND_BACKEND_GRPC_MODEL_EXECUTOR_H_
#define BAND_BACKEND_GRPC_MODEL_EXECUTOR_H_

#include "band/config.h"
#include "band/backend/grpc/model.h"
#include "band/backend/grpc/grpc_client.h"
#include "band/interface/model_executor.h"

namespace band {
namespace grpc {

class GrpcModelExecutor : public interface::IModelExecutor {
 public:
  using interface::IModelExecutor::IModelExecutor;

  GrpcModelExecutor(
      ModelId model_id, WorkerId worker_id, DeviceFlag device_flag,
      std::shared_ptr<BackendConfig> backend_config,
      CpuSet thread_affinity_mask = BandCPUMaskGetSet(CPUMaskFlag::kAll),
      int num_threads = -1)
      : IModelExecutor(model_id, worker_id, device_flag, backend_config,
                       thread_affinity_mask, num_threads) {
    auto grpc_config =
        reinterpret_cast<GrpcBackendConfig*>(backend_config.get());
    client_.Connect(grpc_config->host, grpc_config->port);
  }
  ~GrpcModelExecutor() override;

  absl::StatusOr<ModelSpec> InvestigateModelSpec(
      interface::IModel* model) override;
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
  void ForEachSubgraph(
      std::function<void(const SubgraphKey&)> visitor) override;

 private:
  GrpcClient client_;
  std::map<SubgraphKey, GrpcModel*> model_descriptors_;
};

}  // namespace grpc
}  // namespace band

#endif  // BAND_BACKEND_GRPC_MODEL_EXECUTOR_H_