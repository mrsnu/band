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
#include "tensorflow/lite/interpreter_builder.h"

#if !defined(__ANDROID__) && !defined(__APPLE__) && !defined(_WIN32)
#include <dlfcn.h>
#endif
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tflite_with_xnnpack_optional.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/version.h"

#if defined(TFLITE_ENABLE_DEFAULT_PROFILER)
#include "tensorflow/lite/profiling/platform_profiler.h"
#endif

// aligned_alloc is available (via cstdlib/stdlib.h) with C++17/C11.
#if __cplusplus >= 201703L || __STDC_VERSION__ >= 201112L
#if !defined(__ANDROID__) || __ANDROID_API__ >= 28
// Neither Apple nor Windows provide aligned_alloc.
#if !defined(__APPLE__) && !defined(_WIN32)
#define TFLITE_USE_STD_ALIGNED_ALLOC
#endif
#endif
#endif

namespace tflite {

namespace {

// Ensure that ErrorReporter is non-null.
ErrorReporter* ValidateErrorReporter(ErrorReporter* e) {
  return e ? e : DefaultErrorReporter();
}

template <typename T>
TfLiteStatus Copy(const T* data_ptr, TfLiteIntArray** arr) {
  if (data_ptr->values() == nullptr) {
    return kTfLiteError;
  }

  int size = data_ptr->values()->size();
  *arr = TfLiteIntArrayCreate(size);
  for (int i = 0; i < size; i++) {
    (*arr)->data[i] = static_cast<int>(data_ptr->values()->Get(i));
  }
  return kTfLiteOk;
}

TfLiteStatus ParseSparseIndexVector(const DimensionMetadata* src,
                                    TfLiteDimensionMetadata* tgt) {
  if (src->array_segments() == nullptr || src->array_indices() == nullptr) {
    return kTfLiteError;
  }
  TfLiteStatus status = kTfLiteOk;
  switch (src->array_segments_type()) {
    case SparseIndexVector_Int32Vector:
      status = Copy(src->array_segments_as_Int32Vector(), &tgt->array_segments);
      break;
    case SparseIndexVector_Uint16Vector:
      status =
          Copy(src->array_segments_as_Uint16Vector(), &tgt->array_segments);
      break;
    case SparseIndexVector_Uint8Vector:
      status = Copy(src->array_segments_as_Uint8Vector(), &tgt->array_segments);
      break;
    default:
      status = kTfLiteError;
      break;
  }
  if (status != kTfLiteOk) return status;

  switch (src->array_indices_type()) {
    case SparseIndexVector_Int32Vector:
      return Copy(src->array_indices_as_Int32Vector(), &tgt->array_indices);
    case SparseIndexVector_Uint16Vector:
      return Copy(src->array_indices_as_Uint16Vector(), &tgt->array_indices);
    case SparseIndexVector_Uint8Vector:
      return Copy(src->array_indices_as_Uint8Vector(), &tgt->array_indices);
    default:
      break;
  }
  return kTfLiteError;
}

}  // namespace

const char* kEmptyTensorName = "";

// Using weak symbols to create a delegate allows automatic injection of the
// delegate simply by adding it as a dependency.
// For flex delegate, see also the strong override in
// lite/delegates/flex/delegate.cc.
TFLITE_ATTRIBUTE_WEAK Interpreter::TfLiteDelegatePtr AcquireFlexDelegate() {
#if !defined(__ANDROID__) && !defined(__APPLE__) && !defined(_WIN32)
  // If _pywrap_tensorflow_internal.so is available, use
  // TF_AcquireFlexDelegate() to initialize flex delegate.
  void* lib_tf_internal =
      dlopen("_pywrap_tensorflow_internal.so", RTLD_NOW | RTLD_LOCAL);
  if (lib_tf_internal) {
    auto TF_AcquireFlexDelegate =
        reinterpret_cast<Interpreter::TfLiteDelegatePtr (*)()>(
            dlsym(lib_tf_internal, "TF_AcquireFlexDelegate"));
    if (TF_AcquireFlexDelegate) {
      return TF_AcquireFlexDelegate();
    }
  }
#endif
  return Interpreter::TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

namespace impl {
// TODO(dostos): get error_reporter_ from interpreter
ErrorReporter* InterpreterBuilder::error_reporter_ = DefaultErrorReporter();

TfLiteStatus InterpreterBuilder::BuildLocalIndexToRegistrationMapping(
    const ::tflite::Model* model, const OpResolver& op_resolver) {
  TfLiteStatus status = kTfLiteOk;
  // Reset state.
  flatbuffer_op_index_to_registration_.clear();
  unresolved_custom_ops_.clear();

  auto opcodes = model->operator_codes();
  if (!opcodes) {
    return status;
  }
  int num_custom_ops = 0;
  for (const OperatorCode* opcode : *opcodes) {
    if (opcode->builtin_code() == BuiltinOperator_CUSTOM) {
      num_custom_ops++;
    }
  }
  unresolved_custom_ops_.reserve(num_custom_ops);
  for (const OperatorCode* opcode : *opcodes) {
    const TfLiteRegistration* registration = nullptr;
    status = GetRegistrationFromOpCode(opcode, op_resolver, error_reporter_,
                                       &registration);
    if (status != kTfLiteOk) {
      if (opcode->builtin_code() != BuiltinOperator_CUSTOM) {
        return status;
      }
      // If it's an unresolved custom op, allow it for now. It might be resolved
      // by a delegate later.
      if (!opcode->custom_code()) {
        error_reporter_->Report(
            "Operator with CUSTOM builtin_code has no custom_code.\n");
        return status;
      }
      const auto* op_name = opcode->custom_code()->c_str();
      unresolved_custom_ops_.push_back(CreateUnresolvedCustomOp(op_name));
      registration = &unresolved_custom_ops_.back();
      has_flex_op_ |= IsFlexOp(op_name);
      status = kTfLiteOk;
    }
    flatbuffer_op_index_to_registration_.push_back(registration);
  }
  return status;
}

namespace {
template <class T>
std::vector<int> FlatBufferIntArrayToVector(T* flat_array) {
  // Initialize shape of tensors with null shape. Empty vectors are converted
  // to nullptr for models that are constructed via flatbuffers::Pack.
  if (flat_array == nullptr) {
    return {};
  }
  std::vector<int> ret(flat_array->size());
  for (int i = 0; i < flat_array->size(); i++) {
    ret[i] = flat_array->Get(i);
  }
  return ret;
}

// Used to determine how the op data parsing function creates its working space.
class MallocDataAllocator : public BuiltinDataAllocator {
 public:
  void* Allocate(size_t size, size_t alignment_hint) override {
#ifdef TFLITE_USE_STD_ALIGNED_ALLOC
    // Ensure that alignment is a power of two and a multiple of sizeof(void *)
    // and that size is an integral multiple of alignment.
    size_t used_alignment = std::max(alignment_hint, sizeof(void*));
    size_t used_size =
        ((size + used_alignment - 1) / used_alignment) * used_alignment;
    TFLITE_DCHECK(
        (used_alignment != 0) &&
        ((used_alignment & (used_alignment - 1)) == 0));  // is power-of-two
    return aligned_alloc(used_alignment, used_size);
#else
    return malloc(size);
#endif
  }
  void Deallocate(void* data) override { free(data); }
};

}  // namespace

TfLiteStatus InterpreterBuilder::ParseNodes(
    const ::tflite::Model* model, const OpResolver& op_resolver,
    const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
    Subgraph* subgraph, std::set<int> op_indices) {
  TfLiteStatus status = kTfLiteOk;
  int num_ops = op_indices.size();

  // Reduce the number of redundant allocations
  subgraph->ReserveNodes(num_ops);

  for (int i : op_indices) {
    const auto* op = operators->Get(i);
    int index = op->opcode_index();
    if (index < 0 || index >= flatbuffer_op_index_to_registration_.size()) {
      error_reporter_->Report("Missing registration for opcode_index %d\n",
                              index);
      status = kTfLiteError;
      continue;
    }

    const TfLiteRegistration* registration =
        flatbuffer_op_index_to_registration_[index];
    if (registration == nullptr) {
      error_reporter_->Report("Skipping op for opcode_index %d\n", index);
      status = kTfLiteError;
      continue;
    }

    BuiltinOperator op_type =
        static_cast<BuiltinOperator>(registration->builtin_code);

    if (op_type != BuiltinOperator_CUSTOM && op->custom_options()) {
      error_reporter_->Report(
          "Found builtin operator %s with custom options.\n",
          EnumNameBuiltinOperator(op_type));
    }

    if (op_type == BuiltinOperator_CUSTOM) {
      if (op->custom_options()) {
        subgraph->AddNodeWithParameters(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()),
            FlatBufferIntArrayToVector(op->intermediates()),
            reinterpret_cast<const char*>(op->custom_options()->data()),
            op->custom_options()->size(), nullptr, registration);
      } else {
        subgraph->AddNodeWithParameters(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()),
            FlatBufferIntArrayToVector(op->intermediates()), nullptr, 0,
            nullptr, registration);
      }
    } else {
      void* builtin_data = nullptr;
      MallocDataAllocator malloc_allocator;
      TF_LITE_ENSURE_STATUS(ParseOpData(op, op_type, error_reporter_,
                                        &malloc_allocator, &builtin_data));
      subgraph->AddNodeWithParameters(
          FlatBufferIntArrayToVector(op->inputs()),
          FlatBufferIntArrayToVector(op->outputs()),
          FlatBufferIntArrayToVector(op->intermediates()), nullptr, 0,
          builtin_data, registration);
    }
  }

  return status;
}

TfLiteStatus InterpreterBuilder::ParseQuantization(
    const QuantizationParameters* src_quantization,
    TfLiteQuantization* quantization, const std::vector<int>& dims) {
  quantization->type = kTfLiteNoQuantization;
  if (!src_quantization || !src_quantization->scale() ||
      src_quantization->scale()->size() == 0) {
    return kTfLiteOk;
  }
  if (!src_quantization->zero_point()) {
    error_reporter_->Report(
        "Quantization parameters has non-null scale but null zero_point.");
    return kTfLiteError;
  }

  // Ensure that the number of scales matches the number of zero_points.
  if (src_quantization->scale()->size() !=
      src_quantization->zero_point()->size()) {
    error_reporter_->Report(
        "QuantizationParam has %d zero_point values and %d scale values. Must "
        "have same number.",
        src_quantization->zero_point()->size(),
        src_quantization->scale()->size());
    return kTfLiteError;
  }

  const size_t num_scales = src_quantization->scale()->size();

  // Ensure that the quantization dimension is valid.
  if (src_quantization->quantized_dimension() < 0 ||
      (!dims.empty() &&
       src_quantization->quantized_dimension() >= dims.size())) {
    error_reporter_->Report(
        "quantized_dimension must be in range [0, %d). Was %d.", dims.size(),
        src_quantization->quantized_dimension());
    return kTfLiteError;
  }

  // Ensure that the number of scales is 1 for per-layer quantization, and
  // matches number of quantization dimensions for per-axis quantization.
  if (num_scales != 1 &&
      (!dims.empty() &&
       num_scales != dims[src_quantization->quantized_dimension()])) {
    error_reporter_->Report(
        "num_scales must be 1 for per-layer quantization, or %d for per-axis "
        "quantization, but got %d.",
        dims[src_quantization->quantized_dimension()], num_scales);
    return kTfLiteError;
  }

  // Affine-quantization.
  quantization->type = kTfLiteAffineQuantization;
  auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  affine_quantization->scale = TfLiteFloatArrayCreate(num_scales);
  affine_quantization->zero_point = TfLiteIntArrayCreate(num_scales);
  for (size_t i = 0; i < num_scales; ++i) {
    affine_quantization->scale->data[i] = src_quantization->scale()->Get(i);
    affine_quantization->zero_point->data[i] =
        src_quantization->zero_point()->Get(i);
  }
  affine_quantization->quantized_dimension =
      src_quantization->quantized_dimension();
  quantization->params = reinterpret_cast<void*>(affine_quantization);
  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::ParseSparsity(
    const SparsityParameters* src_sparsity, TfLiteSparsity** sparsity_ptr) {
  if (!src_sparsity) {
    return kTfLiteOk;
  }

  if (src_sparsity->traversal_order() == nullptr ||
      src_sparsity->dim_metadata() == nullptr) {
    error_reporter_->Report("Invalid sparsity parameter.");
    return kTfLiteError;
  }

  auto* sparsity =
      reinterpret_cast<TfLiteSparsity*>(malloc(sizeof(TfLiteSparsity)));
  memset(sparsity, 0, sizeof(TfLiteSparsity));
  *sparsity_ptr = sparsity;

  const size_t traversal_order_size = src_sparsity->traversal_order()->size();
  sparsity->traversal_order = TfLiteIntArrayCreate(traversal_order_size);
  for (int i = 0; i < traversal_order_size; i++) {
    sparsity->traversal_order->data[i] =
        src_sparsity->traversal_order()->Get(i);
  }

  if (src_sparsity->block_map()) {
    const size_t block_map_size = src_sparsity->block_map()->size();
    sparsity->block_map = TfLiteIntArrayCreate(block_map_size);
    for (int i = 0; i < block_map_size; i++) {
      sparsity->block_map->data[i] = src_sparsity->block_map()->Get(i);
    }
  }

  const size_t dim_metadata_size = src_sparsity->dim_metadata()->size();
  sparsity->dim_metadata_size = dim_metadata_size;
  sparsity->dim_metadata = reinterpret_cast<TfLiteDimensionMetadata*>(
      malloc(dim_metadata_size * sizeof(TfLiteDimensionMetadata)));
  memset(sparsity->dim_metadata, 0,
         dim_metadata_size * sizeof(TfLiteDimensionMetadata));

  for (int i = 0; i < dim_metadata_size; i++) {
    const auto* src_metadata = src_sparsity->dim_metadata()->Get(i);
    if (src_metadata->format() != DimensionType_DENSE &&
        src_metadata->format() != DimensionType_SPARSE_CSR) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "The %dth dimension has unknown type: %d.", i,
                           src_metadata->format());
      return kTfLiteError;
    }
    auto* tgt_metadata = &sparsity->dim_metadata[i];

    tgt_metadata->format =
        static_cast<TfLiteDimensionType>(src_metadata->format());

    if (tgt_metadata->format == kTfLiteDimDense) {
      tgt_metadata->dense_size = src_metadata->dense_size();
    } else {
      if (ParseSparseIndexVector(src_metadata, tgt_metadata) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "The %dth sparse dimension has invalid parameters.", i);
        return kTfLiteError;
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::ParseTensors(
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
    Subgraph* subgraph, std::set<int> tensor_indices) {
  TfLiteStatus status = kTfLiteOk;

  // A little helper to get the names of inputs and outputs. Note that they
  // must outlive the subgraph.
  auto get_name = [](const tflite::Tensor* t) -> const char* {
    auto name = t->name();
    if (name) return name->c_str();
    return kEmptyTensorName;
  };

  for (int i = 0; i < tensors->size(); ++i) {
    if (tensor_indices.find(i) == tensor_indices.end()) {
      continue;
    }

    const auto* tensor = tensors->Get(i);
    std::vector<int> dims = FlatBufferIntArrayToVector(tensor->shape());

    TfLiteType type;
    if (ConvertTensorType(tensor->type(), &type, error_reporter_) !=
        kTfLiteOk) {
      status = kTfLiteError;
      continue;
    }

    // TODO(dhkim): delete tensor_types_ from builder attribute.
    // Let model_spec take the control.
    tensor_types_.insert(type);

    auto get_readonly_data = [&](const char** buffer_data,
                                 size_t* buffer_size) {
      // TODO(aselle): Check what happens if we have an unspecified size
      // constant.
      *buffer_data = nullptr;
      if (tensor->buffer() == 0) return kTfLiteOk;
      TF_LITE_ENSURE(error_reporter_, tensor->buffer() < buffers->size());
      if (auto* buffer = (*buffers)[tensor->buffer()]) {
        if (auto* array = buffer->data()) {
          if (size_t size = array->size()) {
            *buffer_size = size;
            *buffer_data = reinterpret_cast<const char*>(array->data());
            return kTfLiteOk;
          }
        }
      }
      return kTfLiteOk;
    };
    size_t buffer_size = 0;
    const char* buffer_ptr;
    TF_LITE_ENSURE_STATUS(get_readonly_data(&buffer_ptr, &buffer_size));

    const auto* src_quantization = tensor->quantization();
    TfLiteQuantization quantization;
    if (ParseQuantization(src_quantization, &quantization, dims) != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "[ParseTensors] Tensor %d has invalid quantization parameters.", i);
      status = kTfLiteError;
    }

    size_t dims_signature_rank = 0;
    const int* dims_signature_data = nullptr;
    if (tensor->shape_signature()) {
      dims_signature_rank = tensor->shape_signature()->size();
      dims_signature_data = tensor->shape_signature()->data();
    }

    bool is_variable = tensor->is_variable();
    if (buffer_ptr) {
      if (is_variable) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "[ParseTensors] Tensor %d is a variable tensor with buffer. "
            "It's not supported now.\n",
            i);
        status = kTfLiteError;
      }

      // TODO(b/144999664): Only constant sparse tensor is supported now.
      const auto* src_sparsity = tensor->sparsity();
      TfLiteSparsity* sparsity = nullptr;
      if (ParseSparsity(src_sparsity, &sparsity) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "[ParseTensors] Tensor %d has invalid sparsity parameters.", i);
        status = kTfLiteError;
      }

      if (subgraph->SetTensorParametersReadOnly(
              i, type, get_name(tensor), dims, quantization, buffer_ptr,
              buffer_size, allocation_, sparsity) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "[ParseTensors] Tensor %d is invalidly specified in schema.\n", i);
        status = kTfLiteError;
      }
    } else {
      if (subgraph->SetTensorParametersReadWrite(
              i, type, get_name(tensor), dims, quantization, is_variable,
              dims_signature_rank, dims_signature_data) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "[ParseTensors] Tensor %d is invalidly specified in schema.\n", i);
        status = kTfLiteError;
      }
    }
  }

  return status;
}

int InterpreterBuilder::AddSubgraph(
    const ::tflite::Model* model, const OpResolver& op_resolver,
    std::unique_ptr<Interpreter>* interpreter, int model_id, int worker_id,
    const std::pair<TfLiteDeviceFlags, std::set<int>>& device_op_indices) {
  auto new_subgraph = CreateSubgraph(model, op_resolver, interpreter, model_id,
                                     worker_id, device_op_indices.second);
  if (new_subgraph == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter_, "[Subgraph] creation failure");
    return -1;
  }

  int subgraph_idx = (*interpreter)->AddSubgraph(std::move(new_subgraph));
  Subgraph* subgraph = (*interpreter)->subgraph(subgraph_idx);

  const SubgraphKey& subgraph_key = subgraph->GetKey();

  TF_LITE_REPORT_ERROR(
      error_reporter_,
      "[Subgraph] added to %dth index for model %d %s from %s to %s",
      subgraph_idx, subgraph_key.model_id,
      TfLiteDeviceGetName(
          (*interpreter)->GetWorkerDeviceFlag(subgraph_key.worker_id)),
      subgraph_key.GetInputOpsString(), subgraph_key.GetOutputOpsString());
  return subgraph_idx;
}

int InterpreterBuilder::num_registered_model = 0;

int InterpreterBuilder::RegisterModel(const FlatBufferModel& model,
                                      ModelConfig* model_config,
                                      const OpResolver& op_resolver,
                                      std::unique_ptr<Interpreter>* interpreter,
                                      int num_threads) {
  return RegisterModel(model.GetModel(), model_config, op_resolver, interpreter,
                       num_threads);
}

int InterpreterBuilder::RegisterModel(const ::tflite::Model* model,
                                      ModelConfig* model_config,
                                      const OpResolver& op_resolver,
                                      std::unique_ptr<Interpreter>* interpreter,
                                      int num_threads) {
  int model_id = (*interpreter)->GetNewModelId();

  int cpu_worker_id = (*interpreter)->GetRepresentativeWorkerId(kTfLiteCPU);
  auto cpu_subgraph =
      CreateSubgraph(model, op_resolver, interpreter, model_id, cpu_worker_id);

  // Add entire model on CPU
  if (cpu_subgraph == nullptr ||
      (*interpreter)->AddSubgraph(std::move(cpu_subgraph)) == -1) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "[Subgraph] Failed to create on CPU delegate");
    (*interpreter)->InvalidateRecentModelId();
    return -1;
  }

  // Create subgraphs
  // Save subgraph_idx - device_op_indices map for prev/next setting
  std::map<int, DeviceOpIndices> subgraph_idx_to_device_ops;

  // Write the ModelSpec for this model
  (*interpreter)->InvestigateModelSpec(model_id);

  ModelSpec& model_spec = (*interpreter)->model_specs_[model_id];

  // TODO(#139): We might generate subgraph indices per `worker_id`
  // to support different op availablity btwn same device types
  // e.g., 2 different NPUs
  // Prepare subgraphs candidates
  const std::string& subgraph_preparation_type =
      (*interpreter)->subgraph_preparation_type_;

  bool need_fallback_subgraph =
    (*interpreter)->GetPlanner()->NeedFallbackSubgraphs() &&
    subgraph_preparation_type != "no_fallback_subgraph";

  // Each pair consists of the unit subgraph index and device op indices.
  std::set<std::pair<int, DeviceOpIndices>> subgraph_indices;
  if ((*interpreter)
          ->GetUnitSubgraphs(model_id, subgraph_indices,
                             need_fallback_subgraph) != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "[Subgraph] Failed to get unit subgraph");
    return -1;
  }

  if (subgraph_preparation_type == "fallback_per_device") {
    // Device,ops to subgraph index map to avoid duplicate
    // subgraph construction without input/output ops
    std::map<DeviceOpIndices, int> device_ops_to_subgraph_index;

    // register subgraphs for all devices
    for (int i = 0; i < kTfLiteNumDevices; ++i) {
      TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);
      std::vector<DeviceOpIndices> device_subgraph_indices =
          (*interpreter)->MakeSubgraphsForFallbackOps(model_id, device_flag);

      for (auto& device_op_indices : device_subgraph_indices) {
        const int worker_id =
            (*interpreter)->GetRepresentativeWorkerId(device_op_indices.first);
        std::string prefix;
        int subgraph_idx = -1;
        // Duplicate subgraph search without key
        if (device_ops_to_subgraph_index.find(device_op_indices) !=
            device_ops_to_subgraph_index.end()) {
          subgraph_idx = device_ops_to_subgraph_index[device_op_indices];
          TFLITE_LOG(INFO) << "[Subgraph] Reuse " << subgraph_idx;
        } else {
          subgraph_idx = AddSubgraph(model, op_resolver, interpreter, model_id,
                                     worker_id, device_op_indices);
          if (subgraph_idx != -1) {
            subgraph_idx_to_device_ops[subgraph_idx] = device_op_indices;
            device_ops_to_subgraph_index[device_op_indices] = subgraph_idx;
          } else {
            continue;
          }
        }

        Subgraph* subgraph = (*interpreter)->subgraph(subgraph_idx);

        if (subgraph == nullptr) {
          TF_LITE_REPORT_ERROR(
              error_reporter_,
              "[Subgraph] Failed to get subgraph from index %d", subgraph_idx);
          continue;
        }

        auto& subgraph_key = subgraph->GetKey();
        // Set unit subgraph indices.
        for (auto& unit_index_device_ops : subgraph_indices) {
          auto unit_index = unit_index_device_ops.first;
          auto& device_ops = unit_index_device_ops.second;
          auto& device = device_ops.first;
          auto& op_indices = device_ops.second;

          if (device == (*interpreter)->GetWorkerDeviceFlag(subgraph_key.worker_id)) {
            if (std::includes(device_op_indices.second.begin(),
                              device_op_indices.second.end(),
                              op_indices.begin(), op_indices.end())) {
              subgraph_key.unit_indices.insert(unit_index);
            }
          }
        }
      }
    }
  } else if (subgraph_preparation_type == "no_fallback_subgraph" ||
             subgraph_preparation_type == "unit_subgraph" ||
             subgraph_preparation_type == "merge_unit_subgraph") {
    // Create subgraphs
    // Save subgraph_idx - device_op_indices map for prev/next setting
    for (auto& subgraph_metadata : subgraph_indices) {
      int unit_subgraph_idx = subgraph_metadata.first;
      auto& device_op_indices = subgraph_metadata.second;

      const int worker_id =
          (*interpreter)->GetRepresentativeWorkerId(device_op_indices.first);
      const int subgraph_idx =
          AddSubgraph(model, op_resolver, interpreter, model_id, worker_id,
                      device_op_indices);
      if (subgraph_idx == -1) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "[Subgraph] Failed to add subgraph to index %d",
                             subgraph_idx);
        continue;
      }
      subgraph_idx_to_device_ops[subgraph_idx] = device_op_indices;

      Subgraph* subgraph = (*interpreter)->subgraph(subgraph_idx);
      if (subgraph == nullptr) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "[Subgraph] Failed to get subgraph from index %d", subgraph_idx);
        continue;
      }

      // Using GetUnitSubgraphs, different from "fallback_per_device",
      // there is no duplicated subgraphs.
      SubgraphKey& subgraph_key = subgraph->GetKey();
      subgraph_key.unit_indices.insert(unit_subgraph_idx);
      TFLITE_LOG(INFO) << "[Subgraph] " << subgraph_idx << "th subgraph has "
                       << unit_subgraph_idx << " unit subgraph.";
    }

    TFLITE_LOG(INFO) << "[Subgraph] " << subgraph_idx_to_device_ops.size()
                     << " subgraphs created during GetUnitSubgraphs()";

    // Add merged atomic subgraphs
    // Note that each merged subgraph consists of unit subgraphs with
    // continuous unit subgraph indices.
    // If we find any of the case that does not satisfy the condition,
    // we should re-implement the merging logic.
    if (subgraph_preparation_type == "merge_unit_subgraph") {
      CreateMergedUnitSubgraphs(model_id, subgraph_idx_to_device_ops, model,
                                op_resolver, interpreter);
    }
  } else {
    
    TF_LITE_REPORT_ERROR(error_reporter_,
                          "[Subgraph] Wrong subgraph_preparation_type %s",
                          subgraph_preparation_type.c_str());
    return -1;
  }

  int num_workers = (*interpreter)->GetNumWorkers();
  std::map<TfLiteDeviceFlags, std::vector<int>> device_to_extra_workers;

  for (int worker_id = 0; worker_id < num_workers; worker_id++) {
    TfLiteDeviceFlags device_flag =
        (*interpreter)->GetWorkerDeviceFlag(worker_id);
    if (worker_id != (*interpreter)->GetRepresentativeWorkerId(device_flag)) {
      device_to_extra_workers[device_flag].push_back(worker_id);
    }
  }

  // Duplicate subgraphs to extra workers
  for (auto& subgraph_metadata : subgraph_idx_to_device_ops) {
    auto device_op_indices = subgraph_metadata.second;
    auto extra_workers_it =
        device_to_extra_workers.find(device_op_indices.first);
    if (extra_workers_it != device_to_extra_workers.end()) {
      for (int extra_worker_id : extra_workers_it->second) {
        const int subgraph_idx =
            AddSubgraph(model, op_resolver, interpreter, model_id, extra_worker_id,
                        device_op_indices);

        if (subgraph_idx == -1) {
          TF_LITE_REPORT_ERROR(error_reporter_,
                               "[Subgraph] Failed to add subgraph to index %d",
                               subgraph_idx);
          continue;
        }

        Subgraph* subgraph = (*interpreter)->subgraph(subgraph_idx);
        if (subgraph == nullptr) {
          TF_LITE_REPORT_ERROR(
              error_reporter_,
              "[Subgraph] Failed to get subgraph from index %d", subgraph_idx);
          continue;
        }

        // Using GetUnitSubgraphs, different from "fallback_per_device",
        // there is no duplicated subgraphs.
        SubgraphKey& subgraph_key = subgraph->GetKey();
        SubgraphKey temp_subgraph_key = subgraph->GetKey();
        temp_subgraph_key.worker_id = (*interpreter)->GetRepresentativeWorkerId(device_op_indices.first);
        Subgraph* representative_subgraph = (*interpreter)->subgraph((*interpreter)->GetSubgraphIdx(temp_subgraph_key));
        subgraph_key.unit_indices = representative_subgraph->GetKey().unit_indices;
      }
    }
  }

  TFLITE_LOG(INFO) << "[Subgraphs] " << (*interpreter)->subgraphs_size()
                   << " subgraphs after duplication for extra workers";

  // Set Prev - Next relation between subgraphs
  std::set<int> model_subgraph_indices;
  for (int i = 0; i < (*interpreter)->subgraphs_size(); i++) {
    Subgraph* subgraph = (*interpreter)->subgraph(i);
    if (subgraph != nullptr && subgraph->GetKey().model_id == model_id) {
      model_subgraph_indices.insert(i);
    }
  }

  for (auto& prev_subgraph_idx : model_subgraph_indices) {
    for (auto& next_subgraph_idx : model_subgraph_indices) {
      // Skip same subgraphs
      if (prev_subgraph_idx == next_subgraph_idx) continue;

      const auto& prev_subgraph_op_indices =
          subgraph_idx_to_device_ops[prev_subgraph_idx];
      const auto& next_subgraph_op_indices =
          subgraph_idx_to_device_ops[next_subgraph_idx];

      // Prev / next subgraphs should not contains common ops
      std::set<int> common_ops;
      std::set_intersection(prev_subgraph_op_indices.second.begin(),
                            prev_subgraph_op_indices.second.end(),
                            next_subgraph_op_indices.second.begin(),
                            next_subgraph_op_indices.second.end(),
                            std::inserter(common_ops, common_ops.begin()));
      if (!common_ops.empty()) continue;

      // Else try to set prev / next subgraphs
      Subgraph* prev_subgraph = (*interpreter)->subgraph(prev_subgraph_idx);
      Subgraph* next_subgraph = (*interpreter)->subgraph(next_subgraph_idx);

      bool is_previous = false;
      
      std::set<int> input_tensors(next_subgraph->inputs().begin(),
                                  next_subgraph->inputs().end());
      for (int tensor_index : prev_subgraph->outputs()) {
        if (input_tensors.find(tensor_index) != input_tensors.end()) {
          is_previous = true;
          break;
        }
      }

      if (is_previous) {
        next_subgraph->SetPrevSubgraph(prev_subgraph);
      }
    }
  }

  if (model_subgraph_indices.size() > 0) {
    if (model_config != nullptr) {
      (*interpreter)->SetModelConfigAndFillProfile(model_id, *model_config);
    }

    if ((*interpreter)->NeedProfile()) {
      (*interpreter)->Profile(model_id);
    }
    return model_id;
  } else {
    (*interpreter)->InvalidateRecentModelId();
    return -1;
  }
}

TfLiteStatus InterpreterBuilder::CreateMergedUnitSubgraphs(
    const int model_id,
    std::map<int, DeviceOpIndices>& subgraph_idx_to_device_ops,
    const ::tflite::Model*& model, const OpResolver& op_resolver,
    std::unique_ptr<Interpreter>* interpreter) {
  Subgraph* primary_subgraph =
      (*interpreter)
          ->subgraph((*interpreter)->GetSubgraphIdx(model_id, kTfLiteCPU));

  // Check all next input tensors are resolved by previous output tensors
  auto is_all_input_prepared =
      [&primary_subgraph](const std::vector<int>& prev_output_tensors,
                          const std::vector<int>& next_input_tensors) {
        for (int input_tensor : next_input_tensors) {
          if (primary_subgraph->tensor(input_tensor)->allocation_type ==
              kTfLiteMmapRo) {
            // parameter tensors are always available,
            // so they always count as "resolved" tensors
            continue;
          }
          if (std::find(prev_output_tensors.begin(), prev_output_tensors.end(),
                        input_tensor) == prev_output_tensors.end()) {
            return false;
          }
        }
        return true;
      };

  // Check given device - op_indices pair is already created or not
  auto is_already_created = [&subgraph_idx_to_device_ops](
                                TfLiteDeviceFlags device,
                                std::set<int> op_indices) {
    for (const auto& idx_device_ops : subgraph_idx_to_device_ops) {
      const std::pair<TfLiteDeviceFlags, std::set<int>>& device_ops =
          idx_device_ops.second;
      if (device_ops.first == device && device_ops.second == op_indices) {
        return true;
      }
    }
    return false;
  };

  int num_subgraphs_before_merge = subgraph_idx_to_device_ops.size();
  bool added = true;
  while (added) {
    added = false;
    std::vector<std::pair<std::set<int>, DeviceOpIndices>> to_add;
    for (const auto& prev_idx_device_ops : subgraph_idx_to_device_ops) {
      for (const auto& next_idx_device_ops : subgraph_idx_to_device_ops) {
        // Skip same subgraph
        if (prev_idx_device_ops.first == next_idx_device_ops.first) continue;
        // Skip different device
        if (prev_idx_device_ops.second.first !=
            next_idx_device_ops.second.first) {
          continue;
        }
        // Skip if there is not resolved output tensor
        Subgraph* prev_subgraph =
            (*interpreter)->subgraph(prev_idx_device_ops.first);
        Subgraph* next_subgraph =
            (*interpreter)->subgraph(next_idx_device_ops.first);
        if (*prev_subgraph->GetKey().unit_indices.rbegin() + 1 !=
            *next_subgraph->GetKey().unit_indices.begin()) {
          continue;
        }
        if (!is_all_input_prepared(prev_subgraph->outputs(),
                                   next_subgraph->inputs())) {
          continue;
        }
        // Prepare merged device - op_indices
        const TfLiteDeviceFlags& device = prev_idx_device_ops.second.first;
        std::set<int> op_indices;
        const std::set<int>& prev_op_indices =
            prev_idx_device_ops.second.second;
        const std::set<int>& next_op_indices =
            next_idx_device_ops.second.second;
        std::set_union(prev_op_indices.begin(), prev_op_indices.end(),
                       next_op_indices.begin(), next_op_indices.end(),
                       std::inserter(op_indices, op_indices.end()));

        std::set<int> unit_subgraph_indices;
        std::set_union(prev_subgraph->GetKey().unit_indices.begin(),
                       prev_subgraph->GetKey().unit_indices.end(),
                       next_subgraph->GetKey().unit_indices.begin(),
                       next_subgraph->GetKey().unit_indices.end(),
                       std::inserter(unit_subgraph_indices, unit_subgraph_indices.end()));
        // Add if not already created
        if (!is_already_created(device, op_indices)) {
          to_add.push_back({unit_subgraph_indices, {device, op_indices}});
        }
      }
    }
    for (auto& subgraph_metadata : to_add) {
      std::set<int>& unit_indices = subgraph_metadata.first;
      const DeviceOpIndices& device_op_indices = subgraph_metadata.second;
 
      const TfLiteDeviceFlags& device_flag = device_op_indices.first;
      const std::set<int>& op_indices = device_op_indices.second;
      if (is_already_created(device_flag, op_indices)) continue;

      int worker_id = (*interpreter)->GetRepresentativeWorkerId(device_flag);
      int subgraph_idx = AddSubgraph(model, op_resolver, interpreter, model_id,
                                     worker_id, device_op_indices);
      if (subgraph_idx == -1) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                              "[Subgraph] Failed to add subgraph to index %d",
                              subgraph_idx);
        return kTfLiteOk;
      }

      added = true;
      subgraph_idx_to_device_ops[subgraph_idx] = device_op_indices;

      Subgraph* subgraph = (*interpreter)->subgraph(subgraph_idx);
      if (subgraph == nullptr) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "[Subgraph] Failed to get subgraph from index %d", subgraph_idx);
        return kTfLiteOk;
      }

      SubgraphKey& subgraph_key = subgraph->GetKey();
      subgraph_key.unit_indices = unit_indices;
    }
  }

  TFLITE_LOG(INFO) << "[Subgraph] "
                   << subgraph_idx_to_device_ops.size() -
                          num_subgraphs_before_merge
                   << " amount of merged subgraph created.";
  return kTfLiteOk;
}

std::unique_ptr<Subgraph> InterpreterBuilder::CreateSubgraph(
    const FlatBufferModel& model, const OpResolver& op_resolver,
    std::unique_ptr<Interpreter>* interpreter, int model_id, int worker_id,
    std::set<int> op_indices, int num_threads) {
  return CreateSubgraph(model.GetModel(), op_resolver, interpreter, model_id,
                        worker_id, op_indices, num_threads);
}

std::unique_ptr<Subgraph> InterpreterBuilder::CreateSubgraph(
    const ::tflite::Model* model, const OpResolver& op_resolver,
    std::unique_ptr<Interpreter>* interpreter, int model_id, int worker_id,
    std::set<int> op_indices, int num_threads) {
  if (!interpreter || !interpreter->get()) {
    TF_LITE_REPORT_ERROR(error_reporter_, "[Subgraph] Interpreter is invalid");
    return {};
  }

  if (!model) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "[Subgraph] Null pointer passed in as model");
    return {};
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "[Subgraph] Model provided is schema version %d not equal "
        "to supported version %d\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return {};
  }

  InterpreterBuilder builder;

  if (builder.BuildLocalIndexToRegistrationMapping(model, op_resolver) !=
      kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter_, "[Subgraph] Registration failed");
    return {};
  }

  // Flatbuffer model schemas define a list of opcodes independent of the graph.
  // We first map those to registrations. This reduces string lookups for custom
  // ops since we only do it once per custom op rather than once per custom op
  // invocation in the model graph.
  // Construct interpreter with correct number of tensors and operators.
  auto* subgraphs = model->subgraphs();
  auto* buffers = model->buffers();

  if (subgraphs->size() == 0) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "[Subgraph] No subgraph in the model");
    return {};
  }

  // Assume FlatBuffer model has only one subgraph.
  // TODO #28: We assume a tflite model has only one Subgraph element
  if (subgraphs->size() > 1) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "[Subgraph] More than one subgraphs in the model");
    return {};
  }

  const tflite::SubGraph* subgraph = (*subgraphs)[0];
  std::unique_ptr<Subgraph> modified_subgraph =
      (*interpreter)->CreateSubgraph();
  auto operators = subgraph->operators();
  auto tensors = subgraph->tensors();
  if (!operators || !tensors || !buffers) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "[Subgraph] Did not get operators, tensors, or buffers in subgraph");
    return {};
  }
  if (modified_subgraph->AddTensors(tensors->size()) != kTfLiteOk) {
    return {};
  }

  if (op_indices.empty()) {
    for (int op_index = 0; op_index <= operators->size() - 1; op_index++) {
      op_indices.insert(op_index);
    }
  }
  // we now parse nodes and tensors, and setup input and
  // output tensors for this particular subgraph

  // first, parse nodes to access `TfLiteNode` info below
  if (builder.ParseNodes(model, op_resolver, operators, modified_subgraph.get(),
                         op_indices) != kTfLiteOk) {
    return {};
  }

  // Collect all input/output tensors for individual nodes.
  // these include intermediate tensors that may be consumed by other
  // nodes in the same model, as well as parameters tensors that aren't
  // really "input" tensors
  std::set<int> node_inputs, node_outputs;
  auto nodes_and_registration = modified_subgraph->nodes_and_registration();
  for (int node_index : modified_subgraph->execution_plan()) {
    TfLiteNode node = nodes_and_registration[node_index].first;
    for (int input_tensor : TfLiteIntArrayView(node.inputs)) {
      node_inputs.insert(input_tensor);
    }
    for (int output_tensor : TfLiteIntArrayView(node.outputs)) {
      node_outputs.insert(output_tensor);
    }
  }

  // merge inputs and outputs to call ParseTensors()
  std::set<int> subgraph_tensors;
  std::set_union(node_inputs.begin(), node_inputs.end(), node_outputs.begin(),
                 node_outputs.end(),
                 std::inserter(subgraph_tensors, subgraph_tensors.end()));

  if (builder.ParseTensors(buffers, tensors, modified_subgraph.get(),
                           subgraph_tensors) != kTfLiteOk) {
    return {};
  }

  // now filter out the intermediate tensors from node_input_tensors so we
  // only have external inputs that are required from outside,
  // as well as parameter tensors
  std::set<int> external_inputs_params;
  std::set_difference(
      node_inputs.begin(), node_inputs.end(), node_outputs.begin(),
      node_outputs.end(),
      std::inserter(external_inputs_params, external_inputs_params.begin()));

  // Next, we need to filter out param tensors from external_inputs_params.
  // there is no way of directly checking if a tensor is a parameter or not,
  // so instead we collect all non-parameter tensors and exclude the param
  // tensors in external_inputs_params that are not in the non-param list
  // NOTE: need to check #65 (Tensor communications between subgraphs)
  std::set<int> non_param_tensors;

  std::vector<int> subgraph_input_vec =
      FlatBufferIntArrayToVector(subgraph->inputs());
  std::set<int> subgraph_inputs =
      std::set<int>(subgraph_input_vec.begin(), subgraph_input_vec.end());
  const std::set<int>& all_node_outputs =
      (*interpreter)->GetModelSpec(model_id).node_output_tensors;
  const std::set<int>& model_outputs =
      (*interpreter)->GetModelSpec(model_id).output_tensors;
  std::set_union(all_node_outputs.begin(), all_node_outputs.end(),
                 subgraph_inputs.begin(), subgraph_inputs.end(),
                 std::inserter(non_param_tensors, non_param_tensors.end()));

  std::set<int> real_inputs;
  std::set_intersection(non_param_tensors.begin(), non_param_tensors.end(),
                        external_inputs_params.begin(),
                        external_inputs_params.end(),
                        std::inserter(real_inputs, real_inputs.begin()));

  std::set<int> real_outputs;
  if (op_indices.size() == operators->size()) {
    // Entire model case doesn't need to consider external nodes
    std::set_difference(node_outputs.begin(), node_outputs.end(),
                        node_inputs.begin(), node_inputs.end(),
                        std::inserter(real_outputs, real_outputs.begin()));
  } else {
    // See if current subgraph outputs model's output tensor
    std::set_intersection(model_outputs.begin(), model_outputs.end(),
                          node_outputs.begin(), node_outputs.end(),
                          std::inserter(real_outputs, real_outputs.begin()));

    // Find reference from external nodes to internal nodes to find real
    // output of current subgraph.
    for (size_t i = 0; i < operators->size(); ++i) {
      // Skip internal nodes
      if (op_indices.find(i) != op_indices.end()) {
        continue;
      }

      const auto* op = operators->Get(i);
      auto op_inputs = FlatBufferIntArrayToVector(op->inputs());

      for (auto external_op_input : op_inputs) {
        if (node_outputs.find(external_op_input) != node_outputs.end()) {
          real_outputs.insert(external_op_input);
        }
      }
    }
  }

  modified_subgraph->SetInputs(
      std::vector<int>(real_inputs.begin(), real_inputs.end()));
  modified_subgraph->SetOutputs(
      std::vector<int>(real_outputs.begin(), real_outputs.end()));

  std::vector<int> variables;
  for (int i = 0; i < modified_subgraph->tensors_size(); ++i) {
    auto* tensor = modified_subgraph->tensor(i);
    if (tensor->is_variable) {
      variables.push_back(i);
    }
  }

  // Find input / output ops
  std::set<int> input_ops;
  std::set<int> output_ops;

  for (int op_index : op_indices) {
    const auto* op = operators->Get(op_index);

    auto input_tensors = FlatBufferIntArrayToVector(op->inputs());
    auto output_tensors = FlatBufferIntArrayToVector(op->outputs());

    for (int input_tensor_index : input_tensors) {
      if (real_inputs.find(input_tensor_index) != real_inputs.end()) {
        input_ops.insert(op_index);
      }
    }

    for (int output_tensor_index : output_tensors) {
      if (real_outputs.find(output_tensor_index) != real_outputs.end()) {
        output_ops.insert(op_index);
      }
    }
  }

  modified_subgraph->SetOpIndices(std::move(op_indices));
  modified_subgraph->SetVariables(std::move(variables));
  modified_subgraph->SetKey(
      SubgraphKey(model_id, worker_id, input_ops, output_ops));

  modified_subgraph->context()->recommended_num_threads = num_threads;

  if ((*interpreter)
          ->ApplyBestDeviceDelegate(
              modified_subgraph.get(),
              (*interpreter)->GetWorkerDeviceFlag(worker_id),
              builder.tensor_types_) != kTfLiteOk)
    return {};

  if (modified_subgraph->AllocateTensors() != kTfLiteOk) {
    return {};
  }

  return std::move(modified_subgraph);
}

}  // namespace impl

}  // namespace tflite
