#include <jni.h>

#include "band/buffer/image_processor.h"
#include "band/java/src/main/native/jni_utils.h"

using band::Buffer;
using band::BufferProcessor;
using band::Tensor;
using namespace band::jni;

extern "C" {
JNIEXPORT void JNICALL Java_org_mrsnu_band_ImageProcessor_process(
    JNIEnv* env, jclass clazz, jlong imageProcessorHandle, jobject bufferObject,
    jobject outputTensorObject) {
  BufferProcessor* processor =
      ConvertLongToBufferProcessor(env, imageProcessorHandle);
  Buffer* buffer = ConvertJObjectToPointer<Buffer>(env, "org/mrsnu/band/Buffer",
                                                   bufferObject);
  Tensor* outputTensor = ConvertJObjectToPointer<Tensor>(
      env, "org/mrsnu/band/Tensor", outputTensorObject);
  if (processor == nullptr || buffer == nullptr || outputTensor == nullptr) {
    // log error about why
    BAND_LOG(band::LogSeverity::kError,
                  "Cannot convert long to object processor: %p, buffer: %p, "
                  "outputTensor: %p",
                  processor, buffer, outputTensor);
    return;
  }

  std::shared_ptr<Buffer> outputTensorBuffer(
      Buffer::CreateFromTensor(outputTensor));

  if (outputTensorBuffer == nullptr) {
    BAND_LOG(band::LogSeverity::kError, "Cannot create buffer from tensor: %p",
                  outputTensor);
    return;
  }

  auto status = processor->Process(*buffer, *outputTensorBuffer);
  if (!status.ok()) {
    // TODO: refactor absl
    BAND_LOG(band::LogSeverity::kError, "Cannot process buffer: %p", buffer);
    return;
  }
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_ImageProcessor_deleteImageProcessor(
    JNIEnv* env, jclass clazz, jlong imageProcessorHandle) {
  delete reinterpret_cast<BufferProcessor*>(imageProcessorHandle);
}
}  // extern "C"