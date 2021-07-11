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
ErrorReporter* InterpreterBuilder::error_reporter_ = DefaultErrorReporter();

TfLiteStatus InterpreterBuilder::BuildLocalIndexToRegistrationMapping(const ::tflite::Model* model,
                                                                      const OpResolver& op_resolver) {
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
    const ::tflite::Model* model,
    const OpResolver& op_resolver,
    const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
    Subgraph* subgraph,
    std::set<int> op_indices) {
  TfLiteStatus status = kTfLiteOk;
  int num_ops = op_indices.size();

  // Reduce the number of redundant allocations
  subgraph->ReserveNodes(num_ops);

  // Operator indices (node index -> op index)
  std::vector<int> node_to_operator_indices;

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

    node_to_operator_indices.push_back(i);
  }

  subgraph->SetOperatorIndices(node_to_operator_indices);

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
    Subgraph* subgraph,
    std::set<int> tensor_indices) {
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
      if (tensor->buffer() >= buffers->size()) {
        error_reporter_->Report(
            "Tensor %d specifies out of range buffer %d (only %d buffers).\n",
            i, tensor->buffer(), buffers->size());
        return kTfLiteError;
      }
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
      error_reporter_->Report("Tensor %d has invalid quantization parameters.",
                              i);
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
        error_reporter_->Report(
            "Tensor %d is a variable tensor with buffer. "
            "It's not supported now.\n",
            i);
        status = kTfLiteError;
      }

      // TODO(b/144999664): Only constant sparse tensor is supported now.
      const auto* src_sparsity = tensor->sparsity();
      TfLiteSparsity* sparsity = nullptr;
      if (ParseSparsity(src_sparsity, &sparsity) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d has invalid sparsity parameters.",
                                i);
        status = kTfLiteError;
      }

      if (subgraph->SetTensorParametersReadOnly(
              i, type, get_name(tensor), dims, quantization, buffer_ptr,
              buffer_size, allocation_, sparsity) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d is invalidly specified in schema.\n",
                                i);
        status = kTfLiteError;
      }
    } else {
      if (subgraph->SetTensorParametersReadWrite(
              i, type, get_name(tensor), dims, quantization, is_variable,
              dims_signature_rank, dims_signature_data) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d is invalidly specified in schema.\n",
                                i);
        status = kTfLiteError;
      }
    }
  }

  return status;
}

int InterpreterBuilder::num_registered_model = 0;

int InterpreterBuilder::RegisterModel(const FlatBufferModel& model,
                     ModelConfig* model_config,
                     const OpResolver& op_resolver,
                     std::unique_ptr<Interpreter>* interpreter,
                     int num_threads) {
  return RegisterModel(
      model.GetModel(), model_config, op_resolver, interpreter, num_threads);
}

int InterpreterBuilder::RegisterModel(const ::tflite::Model* model,
                     ModelConfig* model_config,
                     const OpResolver& op_resolver,
                     std::unique_ptr<Interpreter>* interpreter,
                     int num_threads) {
  int model_id = (*interpreter)->GetNewModelId();
  bool has_available_device = false;

  // the start and end indices aren't valid at this point
  // we fix this later in InvestigateModelSpec
  SubgraphKey subgraph_key(model_id, kTfLiteCPU);
  int subgraph_idx = AddSubgraph(
    model, op_resolver, interpreter, subgraph_key, {}, num_threads);
  if (subgraph_idx != -1) {
    // TODO(dhkim): Move RegisterSubgraphIdx inside AddSubgraph
    (*interpreter)->RegisterSubgraphIdx(subgraph_key, subgraph_idx);
    has_available_device = true;
  }

  // write the ModelSpec for this model
  (*interpreter)->InvestigateModelSpec(model_id);

  ModelSpec& model_spec = (*interpreter)->model_specs_[model_id];

  // register subgraphs for all devices
  for (int i = 0; i < kTfLiteNumDevices; ++i) {
    if (i == kTfLiteCPUFallback) {
      // Skip for Fallback worker
      continue;
    }
    TfLiteDeviceFlags device_id = static_cast<TfLiteDeviceFlags>(i);
    
    std::vector<std::pair<TfLiteDeviceFlags,std::set<int>>>
        subgraph_indices =
        (*interpreter)->MakeSubgraphsForFallbackOps(model_id, device_id);

    Subgraph* previous_subgraph = nullptr;

    for (auto& device_op_indices : subgraph_indices) {
      SubgraphKey key(model_id, device_op_indices.first);
      int subgraph_idx = AddSubgraph(
        model, op_resolver, interpreter, key,
        device_op_indices.second, num_threads);
      if (subgraph_idx != -1) {
        (*interpreter)->RegisterSubgraphIdx(key, subgraph_idx);
        Subgraph* subgraph = (*interpreter)->subgraph(subgraph_idx);
        if (previous_subgraph) {
          if (subgraph->SetPrevSubgraph(
                  previous_subgraph) != kTfLiteOk) {
            TFLITE_LOG(ERROR) << "Failed to set prev subgraph";
            return -1;
          }
        }
        has_available_device = true;
        previous_subgraph = subgraph;
      }

      TFLITE_LOG(INFO) << "ADDED Subgraph "
                       << "Model : " << key.model_id << " "
                       << TfLiteDeviceGetName(key.device_flag) << " "
                       << "From " << key.GetInputOpsString() << " "
                       << "To " << key.GetOutputOpsString();

    }
  }

  if (has_available_device) {
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

int InterpreterBuilder::AddSubgraph(const FlatBufferModel& model,
                                    const OpResolver& op_resolver,
                                    std::unique_ptr<Interpreter>* interpreter,
                                    SubgraphKey& subgraph_key,
                                    std::set<int> op_indices,
                                    int num_threads) {
  return AddSubgraph(model.GetModel(), op_resolver, interpreter,
                     subgraph_key, op_indices, num_threads);
}

int InterpreterBuilder::AddSubgraph(const ::tflite::Model* model,
                                    const OpResolver& op_resolver,
                                    std::unique_ptr<Interpreter>* interpreter,
                                    SubgraphKey& subgraph_key,
                                    std::set<int> op_indices,
                                    int num_threads) {
  int subgraph_exists = (*interpreter)->GetSubgraphIdx(subgraph_key);
  if (subgraph_exists >= 0) {
    return subgraph_exists;
  }

  if (!interpreter || !interpreter->get()) {
    error_reporter_->Report(
        "Interpreter is invalid");
    return -1;
  }

  if (num_threads < -1) {
    error_reporter_->Report(
        "num_threads should be >=0 or just -1 to let TFLite runtime set the "
        "value.");
    return -1;
  }

  if (!model) {
    error_reporter_->Report("Null pointer passed in as model.");
    return -1;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter_->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  InterpreterBuilder builder;

  if (builder.BuildLocalIndexToRegistrationMapping(model, op_resolver) != kTfLiteOk) {
    error_reporter_->Report("Registration failed.\n");
    return -1;
  }

  // Flatbuffer model schemas define a list of opcodes independent of the graph.
  // We first map those to registrations. This reduces string lookups for custom
  // ops since we only do it once per custom op rather than once per custom op
  // invocation in the model graph.
  // Construct interpreter with correct number of tensors and operators.
  auto* subgraphs = model->subgraphs();
  auto* buffers = model->buffers();

  if (subgraphs->size() == 0) {
    error_reporter_->Report("No subgraph in the model.\n");
    return -1;
  }

  // Assume FlatBuffer model has only one subgraph.
  // TODO #28: We assume a tflite model has only one Subgraph element
  if (subgraphs->size() > 1) {
    error_reporter_->Report("More than one subgraphs in the model.\n");
    return -1;
  }

  int old_size = (*interpreter)->subgraphs_size();
  // Only one subgraph will be generated.
  (*interpreter)->AddSubgraphs(subgraphs->size());
  (*interpreter)->SetNumThreads(num_threads, old_size);

  auto cleanup_and_error = [&]() {
    (*interpreter)->DeleteSubgraphs(old_size);
    return -1;
  };

#if defined(TFLITE_ENABLE_DEFAULT_PROFILER)
  (*interpreter)->SetProfiler(tflite::profiling::CreatePlatformProfiler());
#endif

  int modified_subgraph_index = -1;
  for (int subgraph_index = 0; subgraph_index < subgraphs->size();
       ++subgraph_index) {
    const tflite::SubGraph* subgraph = (*subgraphs)[subgraph_index];
    modified_subgraph_index = old_size + subgraph_index;
    tflite::Subgraph* modified_subgraph =
        (*interpreter)->subgraph(modified_subgraph_index);
    auto operators = subgraph->operators();
    auto tensors = subgraph->tensors();
    if (!operators || !tensors || !buffers) {
      error_reporter_->Report(
          "Did not get operators, tensors, or buffers in subgraph %d.\n",
          subgraph_index);
      return cleanup_and_error();
    }
    if (modified_subgraph->AddTensors(tensors->size()) != kTfLiteOk) {
      return cleanup_and_error();
    }

    std::cout << "# of operators " << operators->size() <<
        " # of tensors " << tensors->size() << std::endl;

    if (op_indices.empty()) {
      for (int op_index = 0; op_index <= operators->size() - 1; op_index++) {
        op_indices.insert(op_index);
      }
    }
    // we now parse nodes and tensors, and setup input and
    // output tensors for this particular subgraph

    // first, parse nodes to access `TfLiteNode` info below
    if (builder.ParseNodes(model, op_resolver,
                           operators,
                           modified_subgraph,
                           op_indices) != kTfLiteOk) {
      return cleanup_and_error();
    }

    // Collect all input/output tensors for individual nodes.
    // these include intermediate tensors that may be consumed by other
    // nodes in the same model, as well as parameters tensors that aren't
    // really "input" tensors
    std::set<int> node_inputs, node_outputs;
    std::map<int, int> input_tensor_to_nodes, output_tensor_to_nodes;
    auto nodes_and_registration = modified_subgraph->nodes_and_registration();
    for (int node_index : modified_subgraph->execution_plan()) {
      TfLiteNode node = nodes_and_registration[node_index].first;
      for (int input_tensor : TfLiteIntArrayView(node.inputs)) {
        node_inputs.insert(input_tensor);
        input_tensor_to_nodes[input_tensor] = node_index;
      }
      for (int output_tensor : TfLiteIntArrayView(node.outputs)) {
        node_outputs.insert(output_tensor);
        output_tensor_to_nodes[output_tensor] = node_index;
      }
    }

    // merge inputs and outputs to call ParseTensors()
    std::set<int> subgraph_tensors;
    std::set_union(node_inputs.begin(), node_inputs.end(),
                   node_outputs.begin(), node_outputs.end(),
                   std::inserter(subgraph_tensors, subgraph_tensors.end()));

    if (builder.ParseTensors(buffers, tensors, modified_subgraph,
                             subgraph_tensors) != kTfLiteOk) {
      return cleanup_and_error();
    }

    // now filter out the intermediate tensors from node_input_tensors so we
    // only have external inputs that are required from outside,
    // as well as parameter tensors
    std::set<int> external_inputs_params;
    std::set_difference(node_inputs.begin(), node_inputs.end(),
                        node_outputs.begin(), node_outputs.end(),
                        std::inserter(external_inputs_params,
                                      external_inputs_params.begin()));

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
    std::set<int>& model_outputs =
        (*interpreter)->GetModelSpec(subgraph_key.model_id).node_output_tensors;

    std::set_union(model_outputs.begin(), model_outputs.end(),
                   subgraph_inputs.begin(), subgraph_inputs.end(),
                   std::inserter(non_param_tensors, non_param_tensors.end()));

    std::set<int> real_inputs;
    std::set_intersection(non_param_tensors.begin(), non_param_tensors.end(),
                          external_inputs_params.begin(),
                          external_inputs_params.end(),
                          std::inserter(real_inputs, real_inputs.begin()));

    // we do a similar processing for output tensors, except
    // this time we don't have to worry about param tensors
    std::set<int> real_outputs;
    std::set_difference(node_outputs.begin(), node_outputs.end(),
                        node_inputs.begin(), node_inputs.end(),
                        std::inserter(real_outputs, real_outputs.begin()));

    modified_subgraph->SetInputs(
        std::vector<int>(real_inputs.begin(), real_inputs.end()));
    modified_subgraph->SetOutputs(
        std::vector<int>(real_outputs.begin(), real_outputs.end()));

    modified_subgraph->SetInputTensorToNodes(input_tensor_to_nodes);
    modified_subgraph->SetOutputTensorToNodes(output_tensor_to_nodes);
    std::cout << "Input tensors : " << std::endl;
    for (int i : modified_subgraph->inputs()) {
      std::cout << i << " " << std::endl;
    }

    std::cout << "Input nodes : " << std::endl;
    auto i_nodes = modified_subgraph->input_nodes();
    for (int i :i_nodes) {
      std::cout << modified_subgraph->op_indices()[i] << " " << std::endl;
    }

    std::cout << "Output tensors : " << std::endl;
    for (int i : modified_subgraph->outputs()) {
      std::cout << i << " " << std::endl;
    }

    std::cout << "Output nodes : " << std::endl;
    auto o_nodes = modified_subgraph->output_nodes();
    for (int i : o_nodes) {
      std::cout << modified_subgraph->op_indices()[i] << " " << std::endl;
    }

    std::vector<int> variables;
    for (int i = 0; i < modified_subgraph->tensors_size(); ++i) {
      auto* tensor = modified_subgraph->tensor(i);
      if (tensor->is_variable) {
        variables.push_back(i);
      }
    }

    subgraph_key.input_ops = modified_subgraph->input_ops();
    subgraph_key.output_ops = modified_subgraph->output_ops();

    modified_subgraph->SetVariables(std::move(variables));
    modified_subgraph->SetKey(subgraph_key);

    if ((*interpreter)->
          ApplyBestDeviceDelegate(
            modified_subgraph, 
            subgraph_key.device_flag, 
            builder.tensor_types_) != kTfLiteOk)
      return cleanup_and_error();
    
    if (modified_subgraph->AllocateTensors() != kTfLiteOk)
      return cleanup_and_error();
  }

  return modified_subgraph_index;
}

}  // namespace impl

}  // namespace tflite
