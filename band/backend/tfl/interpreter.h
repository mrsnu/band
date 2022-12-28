#ifndef BAND_BACKEND_TFL_INTERPRETER_H_
#define BAND_BACKEND_TFL_INTERPRETER_H_

#include "band/c/common.h"
#include "band/interface/interpreter.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"

namespace Band {
namespace TfLite {
class TfLiteInterpreter : public Interface::IInterpreter {
 public:
  TfLiteInterpreter() = default;
  ~TfLiteInterpreter() override;

  ModelSpec InvestigateModelSpec(Interface::IModel* model) override;
  // TODO: add a set of ops
  BandStatus FromModel(Interface::IModel* model, WorkerId worker_id,
                       BandDeviceFlags device, std::set<int> ops = {}) override;

  BandBackendType GetBackendType() const override;
  const std::vector<int>& GetInputs(const SubgraphKey& key) const override;
  const std::vector<int>& GetOutputs(const SubgraphKey& key) const override;
  const char* GetInputName(const SubgraphKey& key, int index) const override;
  const char* GetOutputName(const SubgraphKey& key, int index) const override;
  size_t GetNumTensors(const SubgraphKey& key) const override;
  size_t GetNumNodes(const SubgraphKey& key) const override;

  std::shared_ptr<Interface::ITensorView> GetTensorView(const SubgraphKey& key,
                                                        int index) override;

  SubgraphKey GetLargestSubgraphKey(ModelId model_id) const override;
  bool HasSubgraph(const SubgraphKey& key) const override;

  BandStatus InvokeSubgraph(const SubgraphKey& key) override;

 private:
  friend class TfLiteUtil;

  WorkerId worker_id_ = -1;

  tflite::Interpreter* GetInterpreter(const SubgraphKey& key);
  const tflite::Interpreter* GetInterpreter(const SubgraphKey& key) const;

  std::unique_ptr<tflite::Interpreter> CreateTfLiteInterpreter(
      Interface::IModel* model, BandDeviceFlags device,
      std::set<int> op_indices = {});
  std::pair<BandStatus, TfLiteDelegate*> GetDeviceDelegate(
      BandDeviceFlags device);

  std::unordered_map<SubgraphKey, std::unique_ptr<tflite::Interpreter>,
                     SubgraphHash>
      interpreters_;
  std::map<BandDeviceFlags, tflite::Interpreter::TfLiteDelegatePtr> delegates_;
};
}  // namespace TfLite
}  // namespace Band

#endif  // BAND_BACKEND_TFL_INTERPRETER_H_
