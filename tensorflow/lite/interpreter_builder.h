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
/// Deserialization infrastructure for tflite. Provides functionality
/// to go from a serialized tflite model in flatbuffer format to an
/// interpreter.
///
#ifndef TENSORFLOW_LITE_INTERPRETER_BUILDER_H_
#define TENSORFLOW_LITE_INTERPRETER_BUILDER_H_

#include <memory>
#include <set>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace impl {

/// Build an interpreter capable of interpreting `model`.
///
/// model: A model whose lifetime must be at least as long as any
///   interpreter(s) created by the builder. In principle multiple interpreters
///   can be made from a single model.
/// op_resolver: An instance that implements the OpResolver interface, which
/// maps
///   custom op names and builtin op codes to op registrations. The lifetime
///   of the provided `op_resolver` object must be at least as long as the
///   InterpreterBuilder; unlike `model` and `error_reporter`, the `op_resolver`
///   does not need to exist for the duration of any created Interpreter
///   objects.
/// error_reporter: a functor that is called to report errors that handles
///   printf var arg semantics. The lifetime of the `error_reporter` object must
///   be greater than or equal to the Interpreter created by operator().
///
/// Returns a kTfLiteOk when successful and sets interpreter to a valid
/// Interpreter. Note: The user must ensure the model lifetime (and error
/// reporter, if provided) is at least as long as interpreter's lifetime.
class InterpreterBuilder {
 public:
  static void SetErrorReporter(ErrorReporter* error_reporter);

  // Adds a Subgraph to the interpreter.
  // Returns the Subgraph index.
  // Returns -1 if any error occurs.
  static int AddSubgraph(const FlatBufferModel& model,
                        const OpResolver& op_resolver,
                        std::unique_ptr<Interpreter>* interpreter,
                        int num_threads = -1,
                        TfLiteDeviceFlags device_id = kTfLiteCPU);
  static int AddSubgraph(const ::tflite::Model* model,
                     const OpResolver& op_resolver,
                     std::unique_ptr<Interpreter>* interpreter,
                     int num_threads = -1,
                     TfLiteDeviceFlags device_id = kTfLiteCPU);

  // Adds NUM_DEVICES number of Subgraphs to the interpreter.
  // Returns the model id.
  // Returns -1 if any error occurs.
  static int RegisterModel(const FlatBufferModel& model,
                        const OpResolver& op_resolver,
                        std::unique_ptr<Interpreter>* interpreter,
                        int num_threads = -1);
  static int RegisterModel(const ::tflite::Model* model,
                        const OpResolver& op_resolver,
                        std::unique_ptr<Interpreter>* interpreter,
                        int num_threads = -1);

 private:
  InterpreterBuilder() = default;
  ~InterpreterBuilder() = default;

  TfLiteStatus BuildLocalIndexToRegistrationMapping(
      const ::tflite::Model* model, const OpResolver& op_resolver);
  TfLiteStatus ParseNodes(
      const ::tflite::Model* model,
      const OpResolver& op_resolver,
      const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
      Subgraph* subgraph);
  TfLiteStatus ParseTensors(
      const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
      const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
      Subgraph* subgraph);
  TfLiteStatus ParseQuantization(const QuantizationParameters* src_quantization,
                                 TfLiteQuantization* quantization,
                                 const std::vector<int>& dims);
  TfLiteStatus ParseSparsity(const SparsityParameters* src_sparsity,
                             TfLiteSparsity** sparsity);

  static ErrorReporter* error_reporter_;
  static int num_registered_model;

  std::vector<const TfLiteRegistration*> flatbuffer_op_index_to_registration_;
  std::vector<TfLiteRegistration> unresolved_custom_ops_;
  std::vector<BuiltinOperator> flatbuffer_op_index_to_registration_types_;
  const Allocation* allocation_ = nullptr;
  
  bool has_flex_op_ = false;
  std::set<TfLiteType> tensor_types_;
};

}  // namespace impl

}  // namespace tflite

#endif  // TENSORFLOW_LITE_INTERPRETER_BUILDER_H_
