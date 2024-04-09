/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_BACKEND_TFL_MODEL_EXECUTOR_H_
#define BAND_BACKEND_TFL_MODEL_EXECUTOR_H_

#include "band/interface/model_executor.h"
#include "tensorflow/tensorflow/lite/interpreter.h"

namespace band {
namespace tfl {
class TfLiteModelExecutor : public interface::IModelExecutor {
 public:
  using interface::IModelExecutor::IModelExecutor;
  ~TfLiteModelExecutor() override;

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
  friend class TfLiteUtil;

  tflite::Interpreter* GetInterpreter(const SubgraphKey& key);
  const tflite::Interpreter* GetInterpreter(const SubgraphKey& key) const;

  absl::StatusOr<std::unique_ptr<tflite::Interpreter>> CreateTfLiteInterpreter(
      interface::IModel* model, DeviceFlag device,
      std::set<int> op_indices = {});
  static absl::StatusOr<TfLiteDelegate*> GetDeviceDelegate(DeviceFlag device);

  std::unordered_map<SubgraphKey, std::unique_ptr<tflite::Interpreter>,
                     SubgraphHash>
      interpreters_;
  static std::map<DeviceFlag, tflite::Interpreter::TfLiteDelegatePtr>
      delegates_;
};
}  // namespace tfl
}  // namespace band

#endif  // BAND_BACKEND_TFL_MODEL_EXECUTOR_H_
