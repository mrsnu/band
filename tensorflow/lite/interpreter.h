/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Main abstraction controlling the tflite interpreter.
// See context.h for the API for defining operations (TfLiteRegistration).
#ifndef TENSORFLOW_LITE_INTERPRETER_H_
#define TENSORFLOW_LITE_INTERPRETER_H_

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/resource_base.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/type_to_tflitetype.h"
#include "tensorflow/lite/planner/fixed_device_planner.h"
#include "tensorflow/lite/planner/round_robin_planner.h"
#include "tensorflow/lite/planner/shortest_expected_latency_planner.h"
#include "tensorflow/lite/model_builder.h"

#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif

#if TFLITE_EXPERIMENTAL_RUNTIME_EAGER
#include "tensorflow/lite/experimental/tf_runtime/public/eager_interpreter.h"
#endif

namespace tflite {

class InterpreterTest;
class TestDelegate;
namespace delegates {
class InterpreterUtils;  // Class for friend declarations.
}  // namespace delegates

namespace impl {

#define TF_LITE_ENSURE_SUBGRAPH_INDEX(i) \
  do {                                   \
    if (!subgraph(i))                    \
      return kTfLiteError;               \
  } while (0)

/// An interpreter for a graph of nodes that input and output from tensors.
/// Each node of the graph processes a set of input tensors and produces a
/// set of output Tensors. All inputs/output tensors are referenced by index.
///
/// Usage:
///
/// <pre><code>
/// // Create basic model
/// Interpreter foo(2, 1);
/// foo.SetTensorParametersReadWrite(0, ...);
/// foo.SetTensorParametersReadOnly(1, ...);
/// foo.SetNodeParameters(0, ...)
/// // Resize input array to 1 length.
/// foo.ResizeInputTensor(0, 1);
/// foo.AllocateTensors();
/// // Install array data
/// foo.typed_tensor<float>(0)[0] = 3;
/// foo.Invoke();
/// foo.typed_tensor<float>(0)[0] = 4;
/// foo.Invoke();
/// // Resize input array and set data.
/// foo.ResizeInputTensor(0, 2);
/// foo.AllocateTensors();
/// foo.typed_tensor<float>(0)[0] = 4;
/// foo.typed_tensor<float>(0)[1] = 8;
/// foo.Invoke();
/// </code></pre>
///

// a convenient data structure for holding various model information
struct ModelSpec {
  int num_ops;
  std::set<int> output_tensors;
  std::set<TfLiteType> tensor_types;
  std::map<TfLiteDeviceFlags, std::vector<int>> unsupported_ops;
};

class Interpreter {
 public:
  /// Instantiate an interpreter. All errors associated with reading and
  /// processing this model will be forwarded to the error_reporter object.
  //
  /// Note, if error_reporter is nullptr, then a default StderrReporter is
  /// used. Ownership of 'error_reporter' remains with the caller.
  explicit Interpreter(ErrorReporter* error_reporter,
                       TfLitePlannerType planner_type);

  ~Interpreter();

  // Interpreters are not copyable as they have non-trivial memory semantics.
  Interpreter(const Interpreter&) = delete;
  Interpreter& operator=(const Interpreter&) = delete;

  // Functions to build interpreter
#ifndef DOXYGEN_SKIP
  /// Provide a list of tensor indexes that are inputs to the model.
  /// Each index is bound check and this modifies the consistent_ flag of the
  /// interpreter.
  TfLiteStatus SetInputs(size_t subgraph_index, std::vector<int> inputs);

  /// Provide a list of tensor indexes that are outputs to the model
  /// Each index is bound check and this modifies the consistent_ flag of the
  /// interpreter.
  TfLiteStatus SetOutputs(size_t subgraph_index, std::vector<int> outputs);

  /// Provide a list of tensor indexes that are variable tensors.
  /// Each index is bound check and this modifies the consistent_ flag of the
  /// interpreter.
  TfLiteStatus SetVariables(size_t subgraph_index, std::vector<int> variables);

  /// Ensure the internal node storage memory allocates at least `count`
  /// spots for node. NOTE, this doesn't actually add operators. This is an
  /// efficiency optimization that is subject to change.
  void ReserveNodes(size_t subgraph_index, int count);

  /// Adds a node with the given parameters and returns the index of the new
  /// node in `node_index` (optionally). Interpreter will take ownership of
  /// `builtin_data` and destroy it with `free`. Ownership of 'init_data'
  /// remains with the caller.
  TfLiteStatus AddNodeWithParameters(size_t subgraph_index,
                                     const std::vector<int>& inputs,
                                     const std::vector<int>& outputs,
                                     const char* init_data,
                                     size_t init_data_size, void* builtin_data,
                                     const TfLiteRegistration* registration,
                                     int* node_index = nullptr);

  /// Adds `tensors_to_add` tensors, preserving pre-existing Tensor entries.
  /// The value pointed to by `first_new_tensor_index` will be set to the
  /// index of the first new tensor if `first_new_tensor_index` is non-null.
  TfLiteStatus AddTensors(size_t subgraph_index, int tensors_to_add,
                          int* first_new_tensor_index = nullptr);

  /// Set description of inputs/outputs/data/fptrs for node `node_index`.
  /// This variant assumes an external buffer has been allocated of size
  /// bytes. The lifetime of buffer must be ensured to be greater or equal
  /// to Interpreter.
  TfLiteStatus SetTensorParametersReadOnly(
      size_t subgraph_index, size_t tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantization quantization,
      const char* buffer, size_t bytes, const Allocation* allocation = nullptr);

  /// Legacy. Deprecated in favor of above.
  inline TfLiteStatus SetTensorParametersReadOnly(
      size_t subgraph_index, size_t tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantizationParams quantization,
      const char* buffer, size_t bytes,
      const Allocation* allocation = nullptr) {
    return SetTensorParametersReadOnly(subgraph_index, tensor_index, type, name,
                                       dims.size(), dims.data(), quantization,
                                       buffer, bytes, allocation);
  }

  TfLiteStatus SetTensorParametersReadOnly(
      size_t subgraph_index, size_t tensor_index, TfLiteType type, const char* name, 
      const size_t rank, const int* dims, TfLiteQuantizationParams quantization,
      const char* buffer, size_t bytes, const Allocation* allocation = nullptr);

  /// Set description of inputs/outputs/data/fptrs for node `node_index`.
  /// This variant assumes an external buffer has been allocated of size
  /// bytes. The lifetime of buffer must be ensured to be greater or equal
  /// to Interpreter.
  TfLiteStatus SetTensorParametersReadWrite(size_t subgraph_index, size_t tensor_index,
                                            TfLiteType type, const char* name,
                                            const std::vector<int>& dims,
                                            TfLiteQuantization quantization,
                                            bool is_variable = false);

  /// Legacy. Deprecated in favor of above.
  inline TfLiteStatus SetTensorParametersReadWrite(
      size_t subgraph_index, size_t tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantizationParams quantization,
      bool is_variable = false,
      const std::vector<int>* dims_signature = nullptr) {
    size_t rank_dims_signature = 0;
    const int* dims_signature_pointer = nullptr;
    if (dims_signature) {
      rank_dims_signature = dims_signature->size();
      dims_signature_pointer = dims_signature->data();
    }
    return SetTensorParametersReadWrite(
        subgraph_index, tensor_index, type, name, dims.size(), dims.data(),
        quantization, is_variable, rank_dims_signature, dims_signature_pointer);
  }
  TfLiteStatus SetTensorParametersReadWrite(
      size_t subgraph_index, size_t tensor_index, TfLiteType type, const char* name,
      const size_t rank, const int* dims, TfLiteQuantizationParams quantization,
      bool is_variable = false, const size_t rank_dims_signature = 0,
      const int* dims_signature = nullptr);
#endif  // DOXYGEN_SKIP
  // Functions to access tensor data

  /// Read only access to list of inputs.
  const std::vector<int>& inputs(size_t subgraph_index) const {
    assert(subgraph(subgraph_index));
    return subgraphs_[subgraph_index]->inputs();
  }

  /// Return the name of a given input. 
  const char* GetInputName(size_t subgraph_index, size_t index) const {
    if (subgraph_index < subgraphs_.size() && 
        index < inputs(subgraph_index).size()) {
      auto& context = subgraphs_[subgraph_index]->context_;
      return context.tensors[inputs(subgraph_index).at(index)].name;
    } else {
      return nullptr;
    }
  }

  /// Read only access to list of outputs.
  const std::vector<int>& outputs(size_t subgraph_index) const {
    assert(subgraph(subgraph_index));
    return subgraphs_[subgraph_index]->outputs();
  }

  /// Read only access to list of variable tensors.
  const std::vector<int>& variables(size_t subgraph_index) const {
    assert(subgraph(subgraph_index));
    return subgraphs_[subgraph_index]->variables();
  }

  /// Return the name of a given output. 
  const char* GetOutputName(size_t subgraph_index, size_t index) const {
    if (subgraph_index < subgraphs_.size() && 
        index < outputs(subgraph_index).size()) {
      auto& context = subgraphs_[subgraph_index]->context_;
      return context.tensors[outputs(subgraph_index).at(index)].name;
    } else {
      return nullptr;
    }
  }

  /// Return the number of tensors in the model.
  size_t tensors_size(size_t subgraph_index) const {
    assert(subgraph_index < subgraphs_.size());
    return subgraphs_[subgraph_index]->context_.tensors_size;
  }

  /// Return the number of ops in the model.
  size_t nodes_size(size_t subgraph_index) const {
    assert(subgraph_index < subgraphs_.size());
    return subgraphs_[subgraph_index]->nodes_size(); 
  }

  /// WARNING: Experimental interface, subject to change
  const std::vector<int>& execution_plan(size_t subgraph_index) const {
    assert(subgraph(subgraph_index));
    return subgraphs_[subgraph_index]->execution_plan();
  }

#ifndef DOXYGEN_
  /// WARNING: Experimental interface, subject to change
  /// Overrides execution plan. This bounds checks indices sent in.
  TfLiteStatus SetExecutionPlan(size_t subgraph_index, const std::vector<int>& new_plan);
#endif  // DOXYGEN_SKIP

  /// Get a mutable tensor data structure.
  // TODO(aselle): Create a safe ArrayHandle interface to avoid exposing this
  // read/write access to structure
  TfLiteTensor* tensor(size_t subgraph_index, size_t tensor_index) {
    if (subgraph_index < subgraphs_.size() && 
        tensor_index < subgraphs_[subgraph_index]->tensors().size()) {
      return subgraphs_[subgraph_index]->tensor(tensor_index);
    } else {
      return nullptr;
    }
  }

  /// Get an immutable tensor data structure.
  const TfLiteTensor* tensor(size_t subgraph_index, size_t tensor_index) const {
    if (subgraph_index < subgraphs_.size() && 
        tensor_index < subgraphs_[subgraph_index]->tensors().size()) {
      return subgraphs_[subgraph_index]->tensor(tensor_index);
    } else {
      return nullptr;
    }
  }

  /// Get a pointer to an operation and registration data structure.
  const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration(
      size_t subgraph_index, size_t node_index) const {
    // subgraph checks node_index
    if (subgraph_index < subgraphs_.size()) {
      return subgraphs_[subgraph_index]->node_and_registration(node_index);
    } else {
      return nullptr;
    }
  }

  /// Perform a checked cast to the appropriate tensor type (mutable pointer
  /// version).
  template <class T>
  T* typed_tensor(size_t subgraph_index, size_t tensor_index) {
    if (TfLiteTensor* tensor_ptr = tensor(subgraph_index, tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return reinterpret_cast<T*>(tensor_ptr->data.raw);
      }
    }
    return nullptr;
  }

  /// Perform a checked cast to the appropriate tensor type (immutable pointer
  /// version).
  template <class T>
  const T* typed_tensor(size_t subgraph_index, size_t tensor_index) const {
    if (const TfLiteTensor* tensor_ptr = tensor(subgraph_index, tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return reinterpret_cast<const T*>(tensor_ptr->data.raw);
      }
    }
    return nullptr;
  }

  /// Return a mutable pointer to the given input tensor.
  TfLiteTensor* input_tensor(size_t subgraph_index, size_t index) {
    if (index < inputs(subgraph_index).size()) {
      return tensor(subgraph_index, inputs(subgraph_index).at(index)); 
    } else {
      return nullptr;
    }
  }

  /// Return an immutable pointerto the given input tensor.
  const TfLiteTensor* input_tensor(size_t subgraph_index, size_t index) const {
    if (index < inputs(subgraph_index).size()) {
      return tensor(subgraph_index, inputs(subgraph_index).at(index)); 
    } else {
      return nullptr;
    }
  }

  /// Return a mutable pointer into the data of a given input tensor.
  template <class T>
  T* typed_input_tensor(size_t subgraph_index, size_t index) {
    if (index < inputs(subgraph_index).size()) {
      return typed_tensor<T>(subgraph_index, inputs(subgraph_index).at(index));
    } else {
      return nullptr;
    }
  }

  /// Return an immutable pointer into the data of a given input tensor.
  template <class T>
  const T* typed_input_tensor(size_t subgraph_index, size_t index) const {
    if (index < inputs(subgraph_index).size()) {
      return typed_tensor<T>(subgraph_index, inputs(subgraph_index).at(index));
    } else {
      return nullptr;
    }
  }

  /// Return a mutable pointer to the given output tensor.
  TfLiteTensor* output_tensor(size_t subgraph_index, size_t index) {
    if (index < outputs(subgraph_index).size()) {
      return tensor(subgraph_index, outputs(subgraph_index).at(index));
    } else {
      return nullptr;
    }
  }

  /// Return an immutable pointer to the given output tensor.
  const TfLiteTensor* output_tensor(size_t subgraph_index, size_t index) const {
    if (index < outputs(subgraph_index).size()) {
      return tensor(subgraph_index, outputs(subgraph_index).at(index));
    } else {
      return nullptr;
    }
  }

  /// Return a mutable pointer into the data of a given output tensor.
  template <class T>
  T* typed_output_tensor(size_t subgraph_index, size_t index) {
    if (index < outputs(subgraph_index).size()) {
      return typed_tensor<T>(subgraph_index, outputs(subgraph_index).at(index));
    } else {
      return nullptr;
    }
  }

  /// Return an immutable pointer into the data of a given output tensor.
  template <class T>
  const T* typed_output_tensor(size_t subgraph_index, size_t index) const {
    if (index < outputs(subgraph_index).size()) {
      return typed_tensor<T>(subgraph_index, outputs(subgraph_index).at(index));
    } else {
      return nullptr;
    }
  }

  /// Change the dimensionality of a given tensor. Note, this is only acceptable
  /// for tensor indices that are inputs or variables.
  /// Returns status of failure or success. Note that this doesn't actually
  /// resize any existing buffers. A call to AllocateTensors() is required to
  /// change the tensor input buffer.
  TfLiteStatus ResizeInputTensor(size_t subgraph_index, size_t tensor_index,
                                 const std::vector<int>& dims);

  // WARNING: Experimental interface, subject to change
  // Change the dimensionality of a given tensor. This is only acceptable for
  // tensor indices that are inputs or variables. Only unknown dimensions can be
  // resized with this function. Unknown dimensions are indicated as `-1` in the
  // `dims_signature` attribute of a `TfLiteTensor`. Returns status of failure
  // or success.  Note that this doesn't actually resize any existing buffers.
  /// A call to AllocateTensors() is required to change the tensor input buffer.
  TfLiteStatus ResizeInputTensorStrict(size_t subgraph_index, size_t tensor_index,
                                       const std::vector<int>& dims);

  // This releases memory held by non-persistent tensors. It does NOT re-perform
  // memory planning.
  // AllocateTensors needs to be called before next invocation.
  /// WARNING: Experimental interface, subject to change
  TfLiteStatus ReleaseNonPersistentMemory(size_t subgraph_index);

  // Update allocations for all tensors in subgraph that specified in index.
  // This will redim dependent tensors using the input tensor dimensionality as
  // given. This is relatively expensive. This *must be* called after the
  // interpreter has been created and before running inference
  // (and accessing tensor buffers), and *must be* called again if (and only if)
  // an input tensor is resized. Returns status of success or failure.
  TfLiteStatus AllocateTensors();
  TfLiteStatus AllocateTensors(size_t subgraph_index);

  /// Invoke idx-th subgraph in the interpreter.
  /// NOTE: It is possible that the interpreter is not in a ready state
  /// to evaluate (i.e. if a ResizeTensor() has been performed without an
  /// AllocateTensors().
  /// Returns status of success or failure.
  TfLiteStatus Invoke(size_t subgraph_index);

  /// Invoke one subgraph with the model_id in the interpreter.
  /// This method is an asychronous call.
  int InvokeModelAsync(int model_id);
  int InvokeModelAsync(Job request);

  /// Invoke models with a batch size given by the model config.
  /// This method is an asychronous call.
  /// We assume InvokeModelsSync() and InvokeModelsAsync() are
  /// not called consecutively.
  std::vector<int> InvokeModelsAsync();
  std::vector<int> InvokeModelsAsync(std::vector<Job> requests);

  /// Invoke models with a batch size given by the model config.
  /// Returns when all the requests are done.
  /// We assume InvokeModelsSync() and InvokeModelsAsync() are
  /// not called consecutively.
  std::vector<int> InvokeModelsSync();
  std::vector<int> InvokeModelsSync(std::vector<Job> requests);

  // Output subgraph index is valid until next overriding execution.
  std::weak_ptr<int> GetOutputSubgraphIdx(int job_id);

  /// Set the number of threads available to the interpreter.
  ///
  /// NOTE: num_threads should be >= -1.
  /// User may pass -1 to let the TFLite interpreter set the no of threads
  /// available to itself.
  // TODO #7: Change how the interpreter manages context of each subgraph
  void SetNumThreads(int num_threads,
                     size_t first_subgraph_index = 0,
                     int last_subgraph_index = -1);

  void SetXNNPACKNumThreads(int num_threads);

  /// Allow float16 precision for FP32 calculation when possible.
  /// default: not allow.
  /// WARNING: This is an experimental API and subject to change.
  void SetAllowFp16PrecisionForFp32(bool allow);

  /// Get the half precision flag.
  /// WARNING: This is an experimental API and subject to change.
  bool GetAllowFp16PrecisionForFp32() const;

  /// Sets the cancellation function pointer in order to cancel a request in the
  /// middle of a call to Invoke(). The interpreter queries this function during
  /// inference, between op invocations; when it returns true, the interpreter
  /// will abort execution and return `kTfLiteError`. The `data` parameter
  /// contains any data used by the cancellation function, and if non-null,
  /// remains owned by the caller.
  /// WARNING: This is an experimental API and subject to change.
  void SetCancellationFunction(void* data, bool (*check_cancelled_func)(void*));

  // Owning handle to a TfLiteDelegate instance.
  using TfLiteDelegatePtr =
      std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

  /// Ensure the data in `tensor.data` is readable. In case delegate is used,
  /// it might require to copy the data from delegate buffer to raw memory.
  /// WARNING: This is an experimental API and subject to change.
  TfLiteStatus EnsureTensorDataIsReadable(size_t subgraph_index, size_t tensor_index);

  /// Set the delegate buffer handle to a tensor. It can be called in the
  /// following cases:
  /// 1. Set the buffer handle to a tensor that's not being written by a
  ///    delegate. For example, feeding an OpenGL texture as the input of the
  ///    inference graph.
  /// 2. Set the buffer handle to a tensor that uses the same delegate.
  ///    For example, set an OpenGL texture as the output of inference, while
  ///    the node which produces output is an OpenGL delegate node.
  /// WARNING: This is an experimental API and subject to change.
  TfLiteStatus SetBufferHandle(size_t subgraph_index,
                               size_t tensor_index,
                               TfLiteBufferHandle buffer_handle,
                               TfLiteDelegate* delegate);

  /// Get the delegate buffer handle, and the delegate which can process the
  /// buffer handle.
  /// WARNING: This is an experimental API and subject to change.
  TfLiteStatus GetBufferHandle(size_t subgraph_index,
                               size_t tensor_index,
                               TfLiteBufferHandle* buffer_handle,
                               TfLiteDelegate** delegate);

  using ModelDeviceToLatency = std::map<SubgraphKey, int64_t>;

  void Profile(const int num_warm_ups, const int num_runs,
               ModelDeviceToLatency& profiled);

  /// Sets the profiler to tracing execution. The caller retains ownership
  /// of the profiler and must ensure its validity.
  /// WARNING: This is an experimental API and subject to change.
  void SetProfiler(Profiler* profiler);

  /// Same as SetProfiler except this interpreter takes ownership
  /// of the provided profiler.
  /// WARNING: This is an experimental API and subject to change.
  void SetProfiler(std::unique_ptr<Profiler> profiler);

  /// Gets the profiler used for op tracing.
  /// WARNING: This is an experimental API and subject to change.
  Profiler* GetProfiler();

  /// Check whether profiling is required or not.
  bool NeedProfile();

  // The default capacity of `tensors_` vector.
  static constexpr int kTensorsReservedCapacity = 128;
  /// The capacity headroom of `tensors_` vector before calling ops'
  /// `prepare` and `invoke` function. In these functions, it's guaranteed
  /// allocating up to `kTensorsCapacityHeadroom` more tensors won't invalidate
  /// pointers to existing tensors.
  static constexpr int kTensorsCapacityHeadroom = 16;

  /// Set if buffer handle output is allowed.
  //
  /// When using hardware delegation, Interpreter will make the data of output
  /// tensors available in `tensor->data` by default. If the application can
  /// consume the buffer handle directly (e.g. reading output from OpenGL
  /// texture), it can set this flag to false, so Interpreter won't copy the
  /// data from buffer handle to CPU memory. WARNING: This is an experimental
  /// API and subject to change.
  void SetAllowBufferHandleOutput(bool allow_buffer_handle_output) {
    allow_buffer_handle_output_ = allow_buffer_handle_output;
  }

  /// Reset all variable tensors to the default value.
  /// If a variable tensor doesn't have a buffer, reset it to zero.
  /// TODO(b/115961645): Implement - If a variable tensor has a buffer, reset it
  /// to the value of the buffer.
  /// WARNING: This is an experimental API and subject to change.
  TfLiteStatus ResetVariableTensors(size_t subgraph_index);

  /// Retrieve an operator's description of its work, for profiling purposes.
  const char* OpProfilingString(size_t subgraph_index,
                                const TfLiteRegistration& op_reg,
                                const TfLiteNode* node) const;

  // Set the value of an external context. TFLite interpreter doesn't take the
  // memory ownership of this external context 'ctx', and the context should
  // outlive the TFLite interpreter.
  void SetExternalContext(TfLiteExternalContextType type,
                          TfLiteExternalContext* ctx);

#ifndef DOXYGEN_SKIP
  /// Adds `subgraphs_to_add` subgraphs, preserving pre-existing Subgraph
  /// entries. The value pointed to by `first_new_subgraph_index` will be set to
  /// the index of the first new subgraph if `first_new_subgraph_index` is
  /// non-null.
  /// WARNING: This is an experimental API and subject to change.
  void AddSubgraphs(int subgraphs_to_add,
                    int* first_new_subgraph_index = nullptr);

  void DeleteSubgraphs(size_t starting_index_to_delete,
                       int subgraphs_to_delete = -1);

  /// Return the number of subgraphs in the model.
  /// WARNING: This is an experimental API and subject to change.
  size_t subgraphs_size() const { return subgraphs_.size(); }

  /// Get a pointer to a subgraph if in bounds.
  /// WARNING: This is an experimental API and subject to change.
  const Subgraph* subgraph(size_t subgraph_index) const {
    if (subgraph_index < 0 ||
        static_cast<size_t>(subgraph_index) >= subgraphs_size())
      return nullptr;
    return &*subgraphs_[subgraph_index];
  }
  
  Subgraph* subgraph(size_t subgraph_index) {
    if (subgraph_index < 0 ||
        static_cast<size_t>(subgraph_index) >= subgraphs_size())
      return nullptr;
    return &*subgraphs_[subgraph_index];
  }
#endif  // DOXYGEN_SKIP

  TfLiteDelegate* delegates(TfLiteDelegateFlags delegate) {
    auto it = delegates_.find(delegate);
    if (it != delegates_.end())
      return it->second.get();
    else
      return nullptr;
  }

  int GetNumDelegates() {
    return delegates_.size();
  }

  std::shared_ptr<Planner> GetPlanner() {
    return planner_;
  }

  TfLiteStatus PrepareLogging(std::string log_path);

  Worker* GetWorker(int device_idx);
  Worker* GetWorker(TfLiteDeviceFlags device);

  std::map<TfLiteDeviceFlags, std::unique_ptr<Worker>>& GetWorkers() {
    return workers_;
  }

  int GetWorkersSize() {
    return workers_.size();
  }

  // Return all subgraph indices that match the given criteria of model_id,
  // device_id, and op start_idx.
  std::set<int> GetSubgraphIdx(int model_id, TfLiteDeviceFlags device_id,
                               int start_idx);

  // Return the subgraph index for model `model_id` on device `device_idx`.
  // Op start and end indices are assumed to be 0 and num_ops-1, i.e., the
  // whole model.
  int GetSubgraphIdx(int model_id, TfLiteDeviceFlags device_id);
  int GetSubgraphIdx(int model_id, int device_idx);

  // Return the subgraph index that matches the given subgraph_key.
  int GetSubgraphIdx(SubgraphKey subgraph_key);

  void DeleteKey(SubgraphKey subgraph_key);

  std::set<int> models() const;

  void SetModelConfig(int model_id, ModelConfig model_config) {
    model_configs_[model_id] = model_config;
  }

  std::map<int, ModelConfig>& GetModelConfig() {
    return model_configs_;
  }
  
  TfLiteStatus SetWorkerThreadAffinity(const CpuSet& thread_affinity_mask, TfLiteDeviceFlags device_id = kTfLiteNumDevices);

  int64_t GetSubgraphProfileResult(SubgraphKey& key);

  void UpdateProfileResult(const SubgraphKey& key,
                           int64_t new_profile);

  void SetProfileSmoothingConstant(float profile_smoothing_factor) {
    profile_smoothing_factor_ = profile_smoothing_factor;
  }


  ModelSpec& GetModelSpec(int model_id) { return model_specs_[model_id]; }

  int GetWindowSize() const;

  void SetWindowSize(int schedule_window_size);

  void AllowWorkSteal();

  // fill in the ModelSpec for this model
  void InvestigateModelSpec(int model_id);

  // Return a pair of the subgraph idx that leads to the shortest final
  // latency, and that final latency value.
  // Note that the returned subgraph may only cover a subset of the remaining
  // ops, but the latency value is calculated with all subgraphs leading to
  // the final op (of the model) in mind.
  std::pair<int, int64_t>
  GetShortestLatency(int model_id, int start_idx, int64_t start_time,
                     std::map<TfLiteDeviceFlags, int64_t>& device_waiting,
                     TfLiteDeviceFlags preceded_device = kTfLiteNumDevices);

  // Generate explicit subgraphs for fallback ops in `model_id`.
  // Consecutive fallback ops are grouped as one fallback subgraph.
  void MakeSubgraphsForFallbackOps(const int model_id,
                                   const TfLiteDeviceFlags device_flag,
                                   std::vector<SubgraphKey>& splitted_op_range);

  ExternalCpuBackendContext* GetCpuBackendContext() {
    return own_external_cpu_backend_context_.get();
  }

 private:
  friend class InterpreterBuilder;
  friend class tflite::InterpreterTest;
  friend class tflite::TestDelegate;
  friend class tflite::delegates::InterpreterUtils;

  std::shared_ptr<Planner> planner_;
  std::map<TfLiteDeviceFlags, std::unique_ptr<Worker>> workers_;

  // Map structure to find subgraph idx with SubgraphKeys
  std::map<SubgraphKey, int> subgraph_idx_map_;

  void RegisterSubgraphIdx(SubgraphKey subgraph_key, size_t subgraph_index);

  // Applies best delegate from the given device to the subgraph.
  TfLiteStatus ApplyBestDeviceDelegate(Subgraph* subgraph, TfLiteDeviceFlags device, const std::set<TfLiteType>& tensor_types);

  /// Set the value of an external context.
  static void SetExternalContext(struct TfLiteContext* context,
                                 TfLiteExternalContextType type,
                                 TfLiteExternalContext* ctx);

  // Helper function that sets the profiler to all subgraphs.
  void SetSubgraphProfiler(Profiler * profiler);

  // Returns true if delegates have been applied.
  bool HasDelegates(size_t subgraph_index);

  // Returns true if cancellation function returns true.
  bool IsCancelled(size_t subgraph_index);

  // Get the error reporter associated with this interpreter.
  ErrorReporter* error_reporter() { return error_reporter_; }

  // Smoothing constant to update profile result.
  // The smaller profile_smoothing_factor_, the smoother the profile results.
  float profile_smoothing_factor_ = 0.1;

  // The error reporter delegate that tflite will forward queries errors to.
  ErrorReporter* error_reporter_ = nullptr;

  std::map<TfLiteDelegateFlags, TfLiteDelegatePtr> delegates_;

  // Map structure to store profiling results in microseconds of (model_id, device_id)
  std::map<SubgraphKey, int64_t> subgraph_profiling_results_map_;
  // Profiler that has been installed and is owned by this interpreter instance.
  // Useful if client profiler ownership is burdensome.
  std::unique_ptr<Profiler> owned_profiler_;

  // Points to the installed Profiler instance.
  Profiler* installed_profiler_ = nullptr;

  bool allow_buffer_handle_output_ = false;

  // List of active external contexts.
  TfLiteExternalContext* external_contexts_[kTfLiteMaxExternalContexts];

  // The default external cpu backend context. After an TFLite interpreter is
  // initialized, 'external_contexts_[kTfLiteCpuBackendContext]' is set to point
  // to this object. However, if this element value is overwritten via calling
  // 'SetExternalContext(kTfLiteCpuBackendContext, ...)', we will reset this to
  // nullptr if necessary.
  std::unique_ptr<ExternalCpuBackendContext> own_external_cpu_backend_context_;

  // Subgraphs
  std::vector<std::unique_ptr<Subgraph>> subgraphs_;

  // Maps to each model's configuration.
  std::map<int, ModelConfig> model_configs_;

  // Maps to model spec
  std::map<int, ModelSpec> model_specs_;

  TfLitePlannerType planner_type_;
  // A map of resources. Owned by interpreter and shared by multiple subgraphs.
  resource::ResourceMap resources_;

  /* private methods related to subgraph scheduling */
  // divide the given subgraphs into groups that share the same start/end idxs
  // e.g., {(0,10): [1,3], (0,20): [2,4]}
  std::map<std::pair<int, int>, std::vector<int>>
  GroupByStartEndIdx(std::vector<int> subgraph_indices);

  // return subgraph indices for model_id and start_idx,
  // excluding subgraphs on preceded_device
  std::vector<int> GetSubgraphCandidates(int model_id, int start_idx,
                                         TfLiteDeviceFlags preceded_device);

  // return the shortest subgraph out of given subgraphs, when the start time
  // and per-device waiting times are taken into account
  std::pair<int, int64_t>
  GetShortestSubgraphIndex(std::vector<int> subgraph_indices,
                           int64_t start_time,
                           std::map<TfLiteDeviceFlags, int64_t>& device_waiting);

  // Update slo values in model_configs_ according to the worst profiled
  // latency of that model x slo_scale.
  // If slo has already been set, or slo_scale <= 0, then this does nothing.
  // Must be called after the models have been profiled.
  void SetSLOBasedOnProfile();

  // Returns the largest profiled latency of `model_id`, across all devices.
  // Must be called after this model has been profiled.
  int64_t GetWorstDeviceProfileResult(int model_id);
};

}  // namespace impl

#if TFLITE_EXPERIMENTAL_RUNTIME_EAGER
using Interpreter = tflrt::EagerInterpreter;
#else
using Interpreter = impl::Interpreter;
#endif

}  // namespace tflite
#endif  // TENSORFLOW_LITE_INTERPRETER_H_
