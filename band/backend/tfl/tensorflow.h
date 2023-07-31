#ifndef BAND_BACKEND_TFL_TENSORFLOW_H_
#define BAND_BACKEND_TFL_TENSORFLOW_H_

// strip error strings from tensorflow
#define TF_LITE_STRIP_ERROR_STRINGS

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"
#endif  // __ANDROID__
#include "absl/strings/str_format.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

#endif  // BAND_BACKEND_TFL_TENSORFLOW_H_