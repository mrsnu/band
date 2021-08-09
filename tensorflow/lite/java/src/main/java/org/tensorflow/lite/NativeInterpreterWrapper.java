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

package org.tensorflow.lite;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.lite.nnapi.NnApiDelegate;

/**
 * An internal wrapper that wraps native interpreter and controls model execution.
 *
 * <p><b>WARNING:</b> Resources consumed by the {@code NativeInterpreterWrapper} object must be
 * explicitly freed by invoking the {@link #close()} method when the {@code
 * NativeInterpreterWrapper} object is no longer needed.
 */
final class NativeInterpreterWrapper implements AutoCloseable {

  NativeInterpreterWrapper(String jsonPath) {
    TensorFlowLite.init();
    errorHandle = createErrorReporter(ERROR_BUFFER_SIZE);
    interpreterHandle = createInterpreter(errorHandle, jsonPath);
  }

  int registerModel(String modelPath, String modelName) {
    long modelHandle = createModel(modelPath, errorHandle);
    int modelId = registerModel(interpreterHandle, modelHandle, errorHandle, modelName);
    modelHandles.put(modelId, modelHandle);
    return modelId;
  }

  int registerModel(ByteBuffer buffer, String modelName) {
    if (buffer == null
        || (!(buffer instanceof MappedByteBuffer)
        && (!buffer.isDirect() || buffer.order() != ByteOrder.nativeOrder()))) {
      throw new IllegalArgumentException(
          "Model ByteBuffer should be either a MappedByteBuffer of the model file, or a direct "
              + "ByteBuffer using ByteOrder.nativeOrder() which contains bytes of model content.");
    }
    ByteBuffer modelByteBuffer = buffer;
    long modelHandle = createModelWithBuffer(modelByteBuffer, errorHandle);
    int modelId = registerModel(interpreterHandle, modelHandle, errorHandle, modelName);
    if (modelId == -1) {
      throw new IllegalArgumentException("Failed to register model. Invalid model id: " + modelId);
    }
    modelHandles.put(modelId, modelHandle);
    modelByteBuffers.put(modelId, modelByteBuffer);
    return modelId;
  }

  /** Releases resources associated with this {@code NativeInterpreterWrapper}. */
  @Override
  public void close() {
    delete(errorHandle, interpreterHandle);

    modelHandles.forEach((modelId, modelHandle) -> deleteModel(modelHandle));
    tensorHandles.forEach((tensorHandle) -> deleteTensor(tensorHandle));

    errorHandle = 0;
    interpreterHandle = 0;
    
    modelHandles.clear();
    modelByteBuffers.clear();
  }

  /** Sets inputs, runs model inference and returns outputs. */
  void runSync(int[] modelIds, Tensor[][] modelInputs, Tensor[][] modelOutputs, long slo) {
    inferenceDurationNanoseconds = -1;

    long inferenceStartNanos = System.nanoTime();
    int[] jobIds = runAsync(modelIds, modelInputs, slo);
    wait(jobIds, modelOutputs);
    long inferenceDurationNanoseconds = System.nanoTime() - inferenceStartNanos;

    // Only set if the entire operation succeeds.
    this.inferenceDurationNanoseconds = inferenceDurationNanoseconds;
  }

  int[] runAsync(int[] modelIds, Tensor[][] modelInputs, long slo) {
    if (modelIds == null) {
      throw new IllegalArgumentException("Input error: modelIds should not be null.");
    }

    if (modelInputs == null || modelInputs.length != modelIds.length) {
      throw new IllegalArgumentException("Input error: modelInputs should not be null or equal to model count.");
    }

    for (int i = 0; i < modelIds.length; i++) {
      int modelId = modelIds[i];
      if (modelInputs[i] == null || modelInputs[i].length != getInputTensorCount(modelId)) {
        throw new IllegalArgumentException("Input error: Inputs should not be null or equal to input count.");
      }
    }

    long[][] inputHandles = new long[modelIds.length][];

    for (int i = 0; i < modelIds.length; i++) {
      inputHandles[i] = new long[modelInputs[i].length];
      
      for (int j = 0; j < inputHandles[i].length; j++) {
        inputHandles[i][j] = modelInputs[i][j].handle();
      }
    }

    return runAsync(modelIds, inputHandles, interpreterHandle, errorHandle, slo);
  }

  private static native int[] runAsync(int[] modelIds, long[][] inputTensorHandles, long interpreterHandle, long errorHandle, long slo);

  void wait(int[] jobIds, Tensor[][] modelOutputs) {
    if (jobIds == null) {
      throw new IllegalArgumentException("Output error: jobIds should not be null.");
    }

    if (modelOutputs == null || jobIds.length != modelOutputs.length) {
      throw new IllegalArgumentException("Output error: modelOutputs should not be null or equal to job count.");
    }
    long[][] outputHandles = new long[jobIds.length][];

    for (int i = 0; i < jobIds.length; i++) {
      outputHandles[i] = new long[modelOutputs[i].length];

      for (int j = 0; j < outputHandles[i].length; j++) {
        outputHandles[i][j] = modelOutputs[i][j].handle();
      }
    }

    wait(jobIds, outputHandles, interpreterHandle, errorHandle);
  }
  private static native void wait(int[] jobIds, long[][] outputTensorHandles, long interpreterHandle, long errorHandle);

  void setNumThreads(int numThreads) {
    numThreads(interpreterHandle, numThreads);
  }

  void resetVariableTensors(int modelId) {
    resetVariableTensors(interpreterHandle, errorHandle, modelId);
  }

  /**
   * Gets the last inference duration in nanoseconds. It returns null if there is no previous
   * inference run or the last inference run failed.
   */
  Long getLastNativeInferenceDurationNanoseconds() {
    return (inferenceDurationNanoseconds < 0) ? null : inferenceDurationNanoseconds;
  }

  /** Gets the number of input tensors. */
  int getInputTensorCount(int modelId) {
    return getInputCount(interpreterHandle, modelId);
  }

  /**
   * Gets the input {@link Tensor} for the provided input index.
   *
   * @throws IllegalArgumentException if the input index is invalid.
   */
  Tensor allocateInputTensor(int modelId, int index) {
    if (index < 0 || index >= getInputTensorCount(modelId)) {
      throw new IllegalArgumentException("Invalid input Tensor index: " + index);
    }
    long handle = allocateInputTensor(interpreterHandle, modelId, index);
    tensorHandles.add(handle);
    return new Tensor(handle);
  }

  // Allocate new tensor for model's inputIdx th input
  private static native long allocateInputTensor(long interpreterHandle, int modelId, int inputIdx);

  /** Gets the number of output tensors. */
  int getOutputTensorCount(int modelId) {
    return getOutputCount(interpreterHandle, modelId);
  }
  
  /**
   * Gets the output {@link Tensor} for the provided output index.
   *
   * @throws IllegalArgumentException if the output index is invalid.
   */
  Tensor allocateOutputTensor(int modelId, int index) {
    if (index < 0 || index >= getOutputTensorCount(modelId)) {
      throw new IllegalArgumentException("Invalid output Tensor index: " + index);
    }
    
    long handle = allocateOutputTensor(interpreterHandle, modelId, index);
    tensorHandles.add(handle);
    return new Tensor(handle);
  }

  // Allocate new tensor for model's outputIdx th output
  private static native long allocateOutputTensor(long interpreterHandle, int modelId, int outputIdx);

  private static native int getOutputDataType(long interpreterHandle, int modelId, int outputIdx);

  private static final int ERROR_BUFFER_SIZE = 512;

  private long errorHandle;

  private long interpreterHandle;

  private long inferenceDurationNanoseconds = -1;

  private Map<Integer, Long> modelHandles = new HashMap<>();
  private Map<Integer, ByteBuffer> modelByteBuffers = new HashMap<>();
  private List<Long> tensorHandles = new ArrayList<>();

  private static native boolean hasUnresolvedFlexOp(long interpreterHandle);

  private static native int getInputCount(long interpreterHandle, int modelId);

  private static native int getOutputCount(long interpreterHandle, int modelId);

  private static native String[] getInputNames(long interpreterHandle, int modelId);

  private static native String[] getOutputNames(long interpreterHandle, int modelId);

  private static native void numThreads(long interpreterHandle, int numThreads);

  private static native void allowFp16PrecisionForFp32(long interpreterHandle, boolean allow);

  private static native void allowBufferHandleOutput(long interpreterHandle, boolean allow);

  private static native long createErrorReporter(int size);

  private static native long createModel(String modelPathOrBuffer, long errorHandle);

  private static native long createModelWithBuffer(ByteBuffer modelBuffer, long errorHandle);

  private static native long createInterpreter(long errorHandle, String jsonPath);

  private static native int registerModel(long interpreterHandle, long modelHandle, long errorHandle, String modelName);

  private static native void resetVariableTensors(long interpreterHandle, long errorHandle, int modelId);

  private static native void delete(long errorHandle, long interpreterHandle);
  
  private static native void deleteModel(long modelHandle);

  private static native void deleteTensor(long tensorHandle);
}
