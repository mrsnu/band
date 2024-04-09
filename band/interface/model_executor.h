#ifndef BAND_INTERFACE_MODEL_EXECUTOR_H_
#define BAND_INTERFACE_MODEL_EXECUTOR_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/common.h"
#include "band/device/cpu.h"
#include "band/interface/backend.h"
#include "band/interface/model.h"
#include "band/model_spec.h"


namespace band {
namespace interface {
/*
  Model executor for specific <IModel, Worker>
*/

class ITensorView;
/**
 * @brief Interface for executing models.
 * 
 * This interface provides methods for executing models and interacting with their subgraphs.
 * It inherits from the IBackendSpecific interface.
 */
class IModelExecutor : public IBackendSpecific {
 public:
  /**
   * @brief Constructs an IModelExecutor object.
   * 
   * @param model_id The ID of the model.
   * @param worker_id The ID of the worker.
   * @param device_flag The device flag.
   * @param thread_affinity_mask The CPU affinity mask for the threads.
   * @param num_threads The number of threads to use.
   */
  IModelExecutor(
      ModelId model_id, WorkerId worker_id, DeviceFlag device_flag,
      CpuSet thread_affinity_mask = BandCPUMaskGetSet(CPUMaskFlag::kAll),
      int num_threads = -1)
      : model_id_(model_id),
        worker_id_(worker_id),
        device_flag_(device_flag),
        thread_affinity_mask_(thread_affinity_mask),
        num_threads_(num_threads > 0 ? num_threads : -1) {}

  /**
   * @brief Destroys the IModelExecutor object.
   */
  virtual ~IModelExecutor() = default;

  /**
   * @brief Investigates the model specification.
   * 
   * @param model The model to investigate.
   * @return An absl::StatusOr object containing the model specification if successful, or an error status otherwise.
   */
  virtual absl::StatusOr<ModelSpec> InvestigateModelSpec(IModel* model) = 0;

  /**
   * @brief Prepares the subgraph for execution.
   * 
   * @param model The model containing the subgraph.
   * @param ops The set of operations to prepare. If empty, prepares all operations.
   * @param unit_indices The set of unit indices to prepare. If empty, prepares all unit indices.
   * @return An absl::Status object indicating success or failure.
   */
  virtual absl::Status PrepareSubgraph(IModel* model, std::set<int> ops = {},
                                       std::set<int> unit_indices = {}) = 0;

  /**
   * @brief Gets the input tensor indices for a given subgraph key.
   * 
   * @param key The subgraph key.
   * @return A const reference to the vector of input tensor indices.
   */
  virtual const std::vector<int>& GetInputs(const SubgraphKey& key) const = 0;

  /**
   * @brief Gets the output tensor indices for a given subgraph key.
   * 
   * @param key The subgraph key.
   * @return A const reference to the vector of output tensor indices.
   */
  virtual const std::vector<int>& GetOutputs(const SubgraphKey& key) const = 0;

  /**
   * @brief Gets the name of an input tensor for a given subgraph key and index.
   * 
   * @param key The subgraph key.
   * @param index The index of the input tensor.
   * @return A pointer to the name of the input tensor.
   */
  virtual const char* GetInputName(const SubgraphKey& key, int index) const = 0;

  /**
   * @brief Gets the name of an output tensor for a given subgraph key and index.
   * 
   * @param key The subgraph key.
   * @param index The index of the output tensor.
   * @return A pointer to the name of the output tensor.
   */
  virtual const char* GetOutputName(const SubgraphKey& key,
                                    int index) const = 0;

  /**
   * @brief Gets the number of tensors in a given subgraph.
   * 
   * @param key The subgraph key.
   * @return The number of tensors in the subgraph.
   */
  virtual size_t GetNumTensors(const SubgraphKey& key) const = 0;

  /**
   * @brief Gets the number of nodes in a given subgraph.
   * 
   * @param key The subgraph key.
   * @return The number of nodes in the subgraph.
   */
  virtual size_t GetNumNodes(const SubgraphKey& key) const = 0;

  /**
   * @brief Gets a tensor view for a given subgraph key and index.
   * 
   * @param key The subgraph key.
   * @param index The index of the tensor.
   * @return A shared pointer to the tensor view.
   */
  virtual std::shared_ptr<ITensorView> GetTensorView(const SubgraphKey& key,
                                                     int index) = 0;

  /**
   * @brief Checks if a subgraph exists for a given subgraph key.
   * 
   * @param key The subgraph key.
   * @return True if the subgraph exists, false otherwise.
   */
  virtual bool HasSubgraph(const SubgraphKey& key) const = 0;

  /**
   * @brief Gets the largest subgraph key.
   * 
   * @return The largest subgraph key.
   */
  virtual SubgraphKey GetLargestSubgraphKey() const = 0;

  /**
   * @brief Executes a subgraph for a given subgraph key.
   * 
   * @param key The subgraph key.
   * @return An absl::Status object indicating success or failure.
   */
  virtual absl::Status ExecuteSubgraph(const SubgraphKey& key) = 0;

  /**
   * @brief Iterates over all subgraphs and applies a visitor function.
   * 
   * @param visitor The visitor function to apply to each subgraph.
   */
  virtual void ForEachSubgraph(
      std::function<void(const SubgraphKey&)> visitor) = 0;

 protected:
  const ModelId model_id_; /**< The ID of the model. */
  const WorkerId worker_id_; /**< The ID of the worker. */
  const DeviceFlag device_flag_; /**< The device flag. */
  const CpuSet thread_affinity_mask_; /**< The CPU affinity mask for the threads. */
  const int num_threads_; /**< The number of threads to use. */

 private:
  // Disable copy due to complexity
  IModelExecutor(const IModelExecutor&) = delete;
  IModelExecutor(const IModelExecutor&&) = delete;
  IModelExecutor& operator=(const IModelExecutor&) = delete;
  IModelExecutor& operator=(const IModelExecutor&&) = delete;
};
}  // namespace interface
}  // namespace band

#endif