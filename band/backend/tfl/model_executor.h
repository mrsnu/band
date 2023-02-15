#ifndef BAND_BACKEND_TFL_MODEL_EXECUTOR_H_
#define BAND_BACKEND_TFL_MODEL_EXECUTOR_H_

#include "band/c/common.h"
#include "band/interface/model_executor.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"

namespace band {
namespace tfl {
class TfLiteModelExecutor : public interface::IModelExecutor {
 public:
  TfLiteModelExecutor(ModelId model_id, WorkerId worker_id,
                      BandDeviceFlags device_flag);
  ~TfLiteModelExecutor() override;

  ModelSpec InvestigateModelSpec(interface::IModel* model) override;
  BandStatus PrepareSubgraph(interface::IModel* model, std::set<int> ops = {},
                             std::set<int> unit_indices = {}) override;

  BandBackendType GetBackendType() const override;
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

  BandStatus ExecuteSubgraph(const SubgraphKey& key) override;
  void ForEachSubgraph(
      std::function<void(const SubgraphKey&)> iterator) override;

 private:
  friend class TfLiteUtil;

  tflite::Interpreter* GetInterpreter(const SubgraphKey& key);
  const tflite::Interpreter* GetInterpreter(const SubgraphKey& key) const;

  std::unique_ptr<tflite::Interpreter> CreateTfLiteInterpreter(
      interface::IModel* model, BandDeviceFlags device,
      std::set<int> op_indices = {});
  static std::pair<BandStatus, TfLiteDelegate*> GetDeviceDelegate(
      BandDeviceFlags device);

  std::unordered_map<SubgraphKey, std::unique_ptr<tflite::Interpreter>,
                     SubgraphHash>
      interpreters_;
  static std::map<BandDeviceFlags, tflite::Interpreter::TfLiteDelegatePtr>
      delegates_;
};
}  // namespace tfl
}  // namespace band

#endif  // BAND_BACKEND_TFL_MODEL_EXECUTOR_H_
