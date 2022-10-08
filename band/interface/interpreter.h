#ifndef BAND_INTERFACE_INTERPRETER_H_
#define BAND_INTERFACE_INTERPRETER_H_

#include <memory>
#include <vector>

#include "band/common.h"
#include "band/interface/backend.h"

namespace Band {
namespace Interface {
/*
  Interpreter for specific <IModel, processor>
*/

class IModel;
class ITensorView;

class IInterpreter : public IBackendSpecific {
 public:
  IInterpreter() = default;
  virtual ~IInterpreter() = default;
  // TODO: Subgraph generation
  virtual ModelSpec InvestigateModelSpec(IModel* model) = 0;
  virtual absl::StatusOr<SubgraphKey> FromModel(IModel* model,
                                                WorkerId worker_id,
                                                BandDeviceFlags device,
                                                std::set<int> ops = {}) = 0;

  virtual const std::vector<int>& GetInputs(const SubgraphKey& key) const = 0;
  virtual const std::vector<int>& GetOutputs(const SubgraphKey& key) const = 0;
  virtual const char* GetInputName(const SubgraphKey& key, int index) const = 0;
  virtual const char* GetOutputName(const SubgraphKey& key,
                                    int index) const = 0;
  virtual size_t GetNumTensors(const SubgraphKey& key) const = 0;
  virtual size_t GetNumNodes(const SubgraphKey& key) const = 0;
  virtual std::shared_ptr<ITensorView> GetTensorView(const SubgraphKey& key,
                                                     int index) = 0;
  virtual bool HasSubgraph(const SubgraphKey& key) const = 0;
  virtual absl::StatusOr<SubgraphKey> GetModelSubgraphKey(ModelId model_id) const = 0;

  virtual absl::Status InvokeSubgraph(const SubgraphKey& key) = 0;

 protected:
  // WARNING: Every subgraph key must be a valid key. Do not use it for
  // generating invalid key.
  static SubgraphKey CreateSubgraphKey(ModelId model_id, WorkerId worker_id) {
    return SubgraphKey(model_id, worker_id);
  }
  static SubgraphKey CreateSubgraphKey(ModelId model_id, WorkerId worker_id,
                                       std::set<int> input_ops,
                                       std::set<int> output_ops) {
    return SubgraphKey(model_id, worker_id, input_ops, output_ops);
  }

 private:
  // Disable copy due to complexity
  IInterpreter(const IInterpreter&) = delete;
  IInterpreter(const IInterpreter&&) = delete;
  IInterpreter& operator=(const IInterpreter&) = delete;
  IInterpreter& operator=(const IInterpreter&&) = delete;
};
}  // namespace Interface
}  // namespace Band

#endif