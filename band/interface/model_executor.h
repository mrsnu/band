#ifndef BAND_INTERFACE_INTERPRETER_H_
#define BAND_INTERFACE_INTERPRETER_H_

#include <functional>
#include <memory>
#include <vector>

#include "band/common.h"
#include "band/interface/backend.h"
#include "band/interface/model.h"
#include "band/model_spec.h"

namespace Band {
namespace Interface {
/*
  Interpreter for specific <IModel, Worker>
*/

class ITensorView;
class IModelExecutor : public IBackendSpecific {
 public:
  IModelExecutor(ModelId model_id, WorkerId worker_id,
                 BandDeviceFlags device_flag)
      : model_id_(model_id), worker_id_(worker_id), device_flag_(device_flag) {}
  virtual ~IModelExecutor() = default;

  virtual ModelSpec InvestigateModelSpec(IModel* model) = 0;
  virtual BandStatus PrepareSubgraph(IModel* model, std::set<int> ops = {},
                                     std::set<int> unit_indices = {}) = 0;

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
  virtual SubgraphKey GetLargestSubgraphKey() const = 0;

  virtual BandStatus ExecuteSubgraph(const SubgraphKey& key) = 0;
  virtual void IterateSubgraphs(
      std::function<void(const SubgraphKey&)> iterator) = 0;

 protected:
  const ModelId model_id_;
  const WorkerId worker_id_;
  const BandDeviceFlags device_flag_;

 private:
  // Disable copy due to complexity
  IModelExecutor(const IModelExecutor&) = delete;
  IModelExecutor(const IModelExecutor&&) = delete;
  IModelExecutor& operator=(const IModelExecutor&) = delete;
  IModelExecutor& operator=(const IModelExecutor&&) = delete;
};
}  // namespace Interface
}  // namespace Band

#endif