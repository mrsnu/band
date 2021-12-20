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

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.checkerframework.checker.nullness.qual.NonNull;

/**
 * Driver class to drive model inference with TensorFlow Lite.
 *
 * <p>A {@code Interpreter} encapsulates a pre-trained TensorFlow Lite model, in which operations
 * are executed for model inference.
 *
 * <p>For example, if a model takes only one input and returns only one output:
 *
 * <pre>{@code
 * try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
 *   interpreter.run(input, output);
 * }
 * }</pre>
 *
 * <p>If a model takes multiple inputs or outputs:
 *
 * <pre>{@code
 * Object[] inputs = {input0, input1, ...};
 * Map<Integer, Object> map_of_indices_to_outputs = new HashMap<>();
 * FloatBuffer ith_output = FloatBuffer.allocateDirect(3 * 2 * 4);  // Float tensor, shape 3x2x4.
 * ith_output.order(ByteOrder.nativeOrder());
 * map_of_indices_to_outputs.put(i, ith_output);
 * try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
 *   interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
 * }
 * }</pre>
 *
 * <p>If a model takes or produces string tensors:
 *
 * <pre>{@code
 * String[] input = {"foo", "bar"};  // Input tensor shape is [2].
 * String[] output = new String[3][2];  // Output tensor shape is [3, 2].
 * try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
 *   interpreter.runForMultipleInputsOutputs(input, output);
 * }
 * }</pre>
 *
 * <p>Orders of inputs and outputs are determined when converting TensorFlow model to TensorFlowLite
 * model with Toco, as are the default shapes of the inputs.
 *
 * <p>When inputs are provided as (multi-dimensional) arrays, the corresponding input tensor(s) will
 * be implicitly resized according to that array's shape. When inputs are provided as {@link Buffer}
 * types, no implicit resizing is done; the caller must ensure that the {@link Buffer} byte size
 * either matches that of the corresponding tensor, or that they first resize the tensor via {@link
 * #resizeInput()}. Tensor shape and type information can be obtained via the {@link Tensor} class,
 * available via {@link #getInputTensor(int)} and {@link #getOutputTensor(int)}.
 *
 * <p><b>WARNING:</b>Instances of a {@code Interpreter} is <b>not</b> thread-safe. A {@code
 * Interpreter} owns resources that <b>must</b> be explicitly freed by invoking {@link #close()}
 *
 * <p>The TFLite library is built against NDK API 19. It may work for Android API levels below 19,
 * but is not guaranteed.
 */
public final class Interpreter implements AutoCloseable {

  /** An options class for controlling runtime interpreter behavior. */
  public static class Options {
    public Options() {}

    /**
     * Sets the number of threads to be used for ops that support multi-threading. Defaults to a
     * platform-dependent value.
     */
    public Options setNumThreads(int numThreads) {
      this.numThreads = numThreads;
      return this;
    }

    /**
     * Sets whether to allow float16 precision for FP32 calculation when possible. Defaults to false
     * (disallow).
     *
     * @deprecated Prefer using {@link
     *     org.tensorflow.lite.nnapi.NnApiDelegate.Options#setAllowFp16(boolean enable)}.
     */
    @Deprecated
    public Options setAllowFp16PrecisionForFp32(boolean allow) {
      this.allowFp16PrecisionForFp32 = allow;
      return this;
    }

    /**
     * Advanced: Set if buffer handle output is allowed.
     *
     * <p>When a {@link Delegate} supports hardware acceleration, the interpreter will make the data
     * of output tensors available in the CPU-allocated tensor buffers by default. If the client can
     * consume the buffer handle directly (e.g. reading output from OpenGL texture), it can set this
     * flag to false, avoiding the copy of data to the CPU buffer. The delegate documentation should
     * indicate whether this is supported and how it can be used.
     *
     * <p>WARNING: This is an experimental interface that is subject to change.
     */
    public Options setAllowBufferHandleOutput(boolean allow) {
      this.allowBufferHandleOutput = allow;
      return this;
    }

    int numThreads = -1;
    Boolean allowFp16PrecisionForFp32;
    Boolean allowBufferHandleOutput;
  }

  /**
   * Initializes a {@code Interpreter}
   */
  public Interpreter(String jsonPath) {
    wrapper = new NativeInterpreterWrapper(jsonPath);
  }

  /**
   * Registers a model to run inference and a set of custom {@link #Options}.
   *
   * @param modelFile: a file of a pre-trained TF Lite model
   * @param modelName: a name of model to be used for profiling
   * @return an ID of the model to use the {@link #run(int, Object, Object)} method.
   * @throws IllegalArgumentException if {@code modelFile} does not encode a valid TensorFlow Lite
   *     model.
   */
  public int registerModel(@NonNull File modelFile, String fileName) {
    return wrapper.registerModel(modelFile.getAbsolutePath(), fileName);
  }

  /**
   * Registers a model to run inference with a {@code ByteBuffer} of a model file and a set of custom
   * {@link #Options}.
   *
   * The {@code ByteBuffer} can be either a {@link MappedByteBuffer} that memory-maps a model file, or a
   * direct {@link ByteBuffer} of nativeOrder() that contains the bytes content of a model.
   *
   * @param byteBuffer: a byte buffer of a pre-trained TF Lite model
   * @param modelName: a name of model to be used for profiling
   * @return an ID of the model to use the {@link #run(int, Object, Object)} method.
   * @throws IllegalArgumentException if {@code byteBuffer} is not a {@link MappedByteBuffer} nor a
   *     direct {@link Bytebuffer} of nativeOrder.
   */
  public int registerModel(@NonNull ByteBuffer byteBuffer, String fileName) {
    return wrapper.registerModel(byteBuffer, fileName);
  }

  public int runSync(
      int modelId, @NonNull Tensor[] inputs, @NonNull Tensor[] outputs) {
    checkNotClosed();
    return wrapper.runSync(new int[]{modelId}, new Tensor[][]{inputs}, new Tensor[][]{outputs}, 0)[0];
  }

  public int runSync(
      int modelId, @NonNull Tensor[] inputs, @NonNull Tensor[] outputs, long slo) {
    checkNotClosed();
    return wrapper.runSync(new int[]{modelId}, new Tensor[][]{inputs}, new Tensor[][]{outputs}, slo)[0];
  }

  public int[] runSyncMultipleRequests(
    @NonNull int[] modelIds, @NonNull Tensor[][] inputs, @NonNull Tensor[][] outputs) {
    checkNotClosed();
    return wrapper.runSync(modelIds, inputs, outputs, 0);
  }

  public int[] runSyncMultipleRequests(
      @NonNull int[] modelIds, @NonNull Tensor[][] inputs, @NonNull Tensor[][] outputs, long slo) {
    checkNotClosed();
    return wrapper.runSync(modelIds, inputs, outputs, slo);
  }

  public int runAsync(
    int modelId, @NonNull Tensor[] inputs) {
    checkNotClosed();
    return wrapper.runAsync(new int[]{modelId}, new Tensor[][]{inputs}, 0)[0];
  }

  public int runAsync(
      int modelId, @NonNull Tensor[] inputs, long slo) {
    checkNotClosed();
    return wrapper.runAsync(new int[]{modelId}, new Tensor[][]{inputs}, slo)[0];
  }

  public int[] runAsyncMultipleRequests(
    int[] modelIds, @NonNull Tensor[][] inputs) {
    checkNotClosed();
    return wrapper.runAsync(modelIds, inputs, 0);
  }

  public int[] runAsyncMultipleRequests(
      int[] modelIds, @NonNull Tensor[][] inputs, long slo) {
    checkNotClosed();
    return wrapper.runAsync(modelIds, inputs, slo);
  }

  public int wait(int jobId, @NonNull Tensor[] outputs) {
    checkNotClosed();
    return wrapper.wait(new int[]{jobId}, new Tensor[][]{outputs})[0];
  }

  public int[] waitMultipleRequests(@NonNull int[] jobIds, @NonNull Tensor[][] outputs) {
    checkNotClosed();
    return wrapper.wait(jobIds, outputs);
  }

  /** Gets the number of input tensors. */
  public int getInputTensorCount(int modelId) {
    checkNotClosed();
    return wrapper.getInputTensorCount(modelId);
  }

  public Tensor allocateInputTensor(int modelId, int index) {
    checkNotClosed();
    return wrapper.allocateInputTensor(modelId, index);
  }

  /** Gets the number of output Tensors. */
  public int getOutputTensorCount(int modelId) {
    checkNotClosed();
    return wrapper.getOutputTensorCount(modelId);
  }

  public Tensor allocateOutputTensor(int modelId, int index) {
    checkNotClosed();
    return wrapper.allocateOutputTensor(modelId, index);
  }
  
  /**
   * Returns native inference timing.
   *
   * @throws IllegalArgumentException if the model is not initialized by the {@link Interpreter}.
   */
  public Long getLastNativeInferenceDurationNanoseconds() {
    checkNotClosed();
    return wrapper.getLastNativeInferenceDurationNanoseconds();
  }

  /**
   * Sets the number of threads to be used for ops that support multi-threading.
   *
   * @deprecated Prefer using {@link Options#setNumThreads(int)} directly for controlling thread
   *     multi-threading. This method will be removed in a future release.
   */
  @Deprecated
  public void setNumThreads(int numThreads) {
    checkNotClosed();
    wrapper.setNumThreads(numThreads);
  }

  /**
   * Advanced: Resets all variable tensors to the default value.
   *
   * <p>If a variable tensor doesn't have an associated buffer, it will be reset to zero.
   *
   * <p>WARNING: This is an experimental API and subject to change.
   */
  public void resetVariableTensors(int modelId) {
    checkNotClosed();
    wrapper.resetVariableTensors(modelId);
  }

  /** Release resources associated with the {@code Interpreter}. */
  @Override
  public void close() {
    if (wrapper != null) {
      wrapper.close();
      wrapper = null;
    }
  }

  // for Object.finalize, see https://bugs.openjdk.java.net/browse/JDK-8165641
  @SuppressWarnings("deprecation")
  @Override
  protected void finalize() throws Throwable {
    try {
      close();
    } finally {
      super.finalize();
    }
  }

  private void checkNotClosed() {
    if (wrapper == null) {
      throw new IllegalStateException("Internal error: The Interpreter has already been closed.");
    }
  }

  NativeInterpreterWrapper wrapper;
}
