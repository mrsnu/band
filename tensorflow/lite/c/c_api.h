/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_C_C_API_H_
#define TENSORFLOW_LITE_C_C_API_H_

#include <stdarg.h>
#include <stdint.h>

#include "common.h"

// --------------------------------------------------------------------------
/// C API for TensorFlow Lite.
///
/// The API leans towards simplicity and uniformity instead of convenience, as
/// most usage will be by language-specific wrappers. It provides largely the
/// same set of functionality as that of the C++ TensorFlow Lite `Interpreter`
/// API, but is useful for shared libraries where having a stable ABI boundary
/// is important.
///
/// Conventions:
/// * We use the prefix TfLite for everything in the API.
/// * size_t is used to represent byte sizes of objects that are
///   materialized in the address space of the calling process.
/// * int is used as an index into arrays.
///
/// Usage:
/// <pre><code>
/// // Create the model and interpreter options.
/// TfLiteModel* model = TfLiteModelCreateFromFile("/path/to/model.tflite");
/// TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
/// TfLiteInterpreterOptionsSetNumThreads(options, 2);
///
/// // Create the interpreter.
/// TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
///
/// // Allocate tensors and populate the input tensor data.
/// TfLiteInterpreterAllocateTensors(interpreter);
/// TfLiteTensor* input_tensor =
///     TfLiteInterpreterGetInputTensor(interpreter, 0);
/// TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
///                            input.size() * sizeof(float));
///
/// // Execute inference.
/// TfLiteInterpreterInvoke(interpreter);
///
/// // Extract the output tensor data.
/// const TfLiteTensor* output_tensor =
//      TfLiteInterpreterGetOutputTensor(interpreter, 0);
/// TfLiteTensorCopyToBuffer(output_tensor, output.data(),
///                          output.size() * sizeof(float));
///
/// // Dispose of the model and interpreter objects.
/// TfLiteInterpreterDelete(interpreter);
/// TfLiteInterpreterOptionsDelete(options);
/// TfLiteModelDelete(model);

#ifdef SWIG
#define TFL_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TFL_COMPILE_LIBRARY
#define TFL_CAPI_EXPORT __declspec(dllexport)
#else
#define TFL_CAPI_EXPORT __declspec(dllimport)
#endif  // TFL_COMPILE_LIBRARY
#else
#define TFL_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// --------------------------------------------------------------------------
// TfLiteVersion returns a string describing version information of the
// TensorFlow Lite library. TensorFlow Lite uses semantic versioning.
TFL_CAPI_EXPORT extern const char* TfLiteVersion(void);

// --------------------------------------------------------------------------
// TfLiteModel wraps a loaded TensorFlow Lite model.
typedef struct TfLiteModel TfLiteModel;

// Returns a model from the provided buffer, or null on failure.
TFL_CAPI_EXPORT extern TfLiteModel* TfLiteModelCreate(const void* model_data,
                                                      size_t model_size);

// Returns a model from the provided file, or null on failure.
TFL_CAPI_EXPORT extern TfLiteModel* TfLiteModelCreateFromFile(
    const char* model_path);

// Destroys the model instance.
TFL_CAPI_EXPORT extern void TfLiteModelDelete(TfLiteModel* model);

// --------------------------------------------------------------------------
// TfLiteInterpreterOptions allows customized interpreter configuration.
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;

// Returns a new interpreter options instances.
TFL_CAPI_EXPORT extern TfLiteInterpreterOptions*
TfLiteInterpreterOptionsCreate();

// Destroys the interpreter options instance.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsDelete(
    TfLiteInterpreterOptions* options);

// Sets a custom error reporter for interpreter execution.
//
// * `reporter` takes the provided `user_data` object, as well as a C-style
//   format string and arg list (see also vprintf).
// * `user_data` is optional. If provided, it is owned by the client and must
//   remain valid for the duration of the interpreter lifetime.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetErrorReporter(
    TfLiteInterpreterOptions* options,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data);

TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetConfigPath(
    TfLiteInterpreterOptions* options,
    const char* config_path);

// --------------------------------------------------------------------------
// TfLiteInterpreter provides inference from a provided model.
typedef struct TfLiteInterpreter TfLiteInterpreter;

// Returns a new interpreter using the provided model and options, or null on
// failure.
//
// * `model` must be a valid model instance. The caller retains ownership of the
//   object, and can destroy it immediately after creating the interpreter; the
//   interpreter will maintain its own reference to the underlying model data.
// * `optional_options` may be null. The caller retains ownership of the object,
//   and can safely destroy it immediately after creating the interpreter.
//
// NOTE: The client *must* explicitly allocate tensors before attempting to
// access input tensor data or invoke the interpreter.
TFL_CAPI_EXPORT extern TfLiteInterpreter* TfLiteInterpreterCreate(const TfLiteInterpreterOptions* optional_options);

// Destroys the interpreter.
TFL_CAPI_EXPORT extern void TfLiteInterpreterDelete(
    TfLiteInterpreter* interpreter);

TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterRegisterModel(TfLiteInterpreter* interpreter, TfLiteModel* model);

TFL_CAPI_EXPORT extern void TfLiteInterpreterInvokeSync(
    TfLiteInterpreter* interpreter, int32_t model_id, TfLiteTensor** inputs, TfLiteTensor** outputs);

TFL_CAPI_EXPORT extern int TfLiteInterpreterInvokeAsync(
    TfLiteInterpreter* interpreter, int32_t model_id, TfLiteTensor** inputs);

TFL_CAPI_EXPORT extern TfLiteStatus TFLiteInterpreterWait(TfLiteInterpreter* interpreter, int job_id, TfLiteTensor** outputs);

// Returns the number of input tensors associated with the model.
TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterGetInputTensorCount(
    const TfLiteInterpreter* interpreter, int32_t model_id);

// Returns the number of output tensors associated with the model.
TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterGetOutputTensorCount(
    const TfLiteInterpreter* interpreter, int32_t model_id);

TFL_CAPI_EXPORT extern TfLiteTensor* TfLiteInterpreterAllocateInputTensor(
    const TfLiteInterpreter* interpreter, int32_t model_id, int32_t input_index);

TFL_CAPI_EXPORT extern TfLiteTensor* TfLiteInterpreterAllocateOutputTensor(
    const TfLiteInterpreter* interpreter, int32_t model_id, int32_t output_index);

TFL_CAPI_EXPORT extern void TfLiteInterpreterDeleteTensor(TfLiteTensor* tensor);

// Returns the type of a tensor element.
TFL_CAPI_EXPORT extern TfLiteType TfLiteTensorType(const TfLiteTensor* tensor);

// Returns the number of dimensions that the tensor has.
TFL_CAPI_EXPORT extern int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor);

// Returns the length of the tensor in the "dim_index" dimension.
// REQUIRES: 0 <= dim_index < TFLiteTensorNumDims(tensor)
TFL_CAPI_EXPORT extern int32_t TfLiteTensorDim(const TfLiteTensor* tensor,
                                               int32_t dim_index);

// Returns the size of the underlying data in bytes.
TFL_CAPI_EXPORT extern size_t TfLiteTensorByteSize(const TfLiteTensor* tensor);

// Returns a pointer to the underlying data buffer.
//
// NOTE: The result may be null if tensors have not yet been allocated, e.g.,
// if the Tensor has just been created or resized and `TfLiteAllocateTensors()`
// has yet to be called, or if the output tensor is dynamically sized and the
// interpreter hasn't been invoked.
TFL_CAPI_EXPORT extern void* TfLiteTensorData(const TfLiteTensor* tensor);

// Returns the (null-terminated) name of the tensor.
TFL_CAPI_EXPORT extern const char* TfLiteTensorName(const TfLiteTensor* tensor);

// Returns the parameters for asymmetric quantization. The quantization
// parameters are only valid when the tensor type is `kTfLiteUInt8` and the
// `scale != 0`. Quantized values can be converted back to float using:
//    real_value = scale * (quantized_value - zero_point);
TFL_CAPI_EXPORT extern TfLiteQuantizationParams TfLiteTensorQuantizationParams(
    const TfLiteTensor* tensor);

// Copies from the provided input buffer into the tensor's buffer.
// REQUIRES: input_data_size == TfLiteTensorByteSize(tensor)
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteTensorCopyFromBuffer(
    TfLiteTensor* tensor, const void* input_data, size_t input_data_size);

// Copies to the provided output buffer from the tensor's buffer.
// REQUIRES: output_data_size == TfLiteTensorByteSize(tensor)
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteTensorCopyToBuffer(
    const TfLiteTensor* output_tensor, void* output_data,
    size_t output_data_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_C_C_API_H_
