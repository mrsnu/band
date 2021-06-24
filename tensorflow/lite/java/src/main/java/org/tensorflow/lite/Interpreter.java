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

    /** Sets whether to use NN API (if available) for op execution. Defaults to false (disabled). */
    public Options setUseNNAPI(boolean useNNAPI) {
      this.useNNAPI = useNNAPI;
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
     * Adds a {@link Delegate} to be applied during interpreter creation.
     *
     * <p>WARNING: This is an experimental interface that is subject to change.
     */
    public Options addDelegate(Delegate delegate) {
      delegates.add(delegate);
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

    /**
     * Experimental: Enable an optimized set of floating point CPU kernels (provided by XNNPACK).
     *
     * <p>Enabling this flag will enable use of a new, highly optimized set of CPU kernels provided
     * via the XNNPACK delegate. Currently, this is restricted to a subset of floating point
     * operations. Eventually, we plan to enable this by default, as it can provide significant
     * peformance benefits for many classes of floating point models. See
     * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md
     * for more details.
     *
     * <p>Things to keep in mind when enabling this flag:
     *
     * <ul>
     *   <li>Startup time and resize time may increase.
     *   <li>Baseline memory consumption may increase.
     *   <li>Compatibility with other delegates (e.g., GPU) has not been fully validated.
     *   <li>Quantized models will not see any benefit.
     * </ul>
     *
     * <p>WARNING: This is an experimental interface that is subject to change.
     */
    public Options setUseXNNPACK(boolean useXNNPACK) {
      this.useXNNPACK = useXNNPACK;
      return this;
    }

    int numThreads = -1;
    Boolean useNNAPI;
    Boolean allowFp16PrecisionForFp32;
    Boolean allowBufferHandleOutput;
    Boolean useXNNPACK;
    final List<Delegate> delegates = new ArrayList<>();
  }

  /**
   * Initializes a {@code Interpreter}
   */
  public Interpreter() {
    wrapper = new NativeInterpreterWrapper();
  }

  /**
   * Registers a model to run inference and a set of custom {@link #Options}.
   *
   * @param modelFile: a file of a pre-trained TF Lite model
   * @param options: a set of options for customizing interpreter behavior
   * @return an ID of the model to use the {@link #run(int, Object, Object)} method.
   * @throws IllegalArgumentException if {@code modelFile} does not encode a valid TensorFlow Lite
   *     model.
   */
  public int registerModel(@NonNull File modelFile, Options options) {
    return wrapper.registerModel(modelFile.getAbsolutePath(), options);
  }

  /**
   * Registers a model to run inference with a {@code ByteBuffer} of a model file and a set of custom
   * {@link #Options}.
   *
   * The {@code ByteBuffer} can be either a {@link MappedByteBuffer} that memory-maps a model file, or a
   * direct {@link ByteBuffer} of nativeOrder() that contains the bytes content of a model.
   *
   * @param byteBuffer: a byte buffer of a pre-trained TF Lite model
   * @param options: a set of options for customizing interpreter behavior
   * @return an ID of the model to use the {@link #run(int, Object, Object)} method.
   * @throws IllegalArgumentException if {@code byteBuffer} is not a {@link MappedByteBuffer} nor a
   *     direct {@link Bytebuffer} of nativeOrder.
   */
  public int registerModel(@NonNull ByteBuffer byteBuffer, Options options) {
    return wrapper.registerModel(byteBuffer, options);
  }

  /**
   * Runs model inference if the model takes only one input, and provides only one output.
   *
   * <p>Warning: The API is more efficient if a {@link Buffer} (preferably direct, but not required)
   * is used as the input/output data type. Please consider using {@link Buffer} to feed and fetch
   * primitive data for better performance. The following concrete {@link Buffer} types are
   * supported:
   *
   * <ul>
   *   <li>{@link ByteBuffer} - compatible with any underlying primitive Tensor type.
   *   <li>{@link FloatBuffer} - compatible with float Tensors.
   *   <li>{@link IntBuffer} - compatible with int32 Tensors.
   *   <li>{@link LongBuffer} - compatible with int64 Tensors.
   * </ul>
   *
   * @param modelId an ID of the target model to run inference
   * @param input an array or multidimensional array, or a {@link Buffer} of primitive types
   *     including int, float, long, and byte. {@link Buffer} is the preferred way to pass large
   *     input data for primitive types, whereas string types require using the (multi-dimensional)
   *     array input path. When a {@link Buffer} is used, its content should remain unchanged until
   *     model inference is done, and the caller must ensure that the {@link Buffer} is at the
   *     appropriate read position. A {@code null} value is allowed only if the caller is using a
   *     {@link Delegate} that allows buffer handle interop, and such a buffer has been bound to the
   *     input {@link Tensor}.
   * @throws IllegalArgumentException if {@code input} or {@code output} is null or empty, or if
   *     error occurs when running the inference.
   */
  public Tensor[] run(int modelId, Tensor input) {
    Tensor[] inputs = {input};
    return runForMultipleInputs(modelId, inputs);
  }

  /**
   * Runs model inference if the model takes multiple inputs, or returns multiple outputs.
   *
   * <p>Warning: The API is more efficient if {@link Buffer}s (preferably direct, but not required)
   * are used as the input/output data types. Please consider using {@link Buffer} to feed and fetch
   * primitive data for better performance. The following concrete {@link Buffer} types are
   * supported:
   *
   * <ul>
   *   <li>{@link ByteBuffer} - compatible with any underlying primitive Tensor type.
   *   <li>{@link FloatBuffer} - compatible with float Tensors.
   *   <li>{@link IntBuffer} - compatible with int32 Tensors.
   *   <li>{@link LongBuffer} - compatible with int64 Tensors.
   * </ul>
   *
   * <p>Note: {@code null} values for invididual elements of {@code inputs} and {@code outputs} is
   * allowed only if the caller is using a {@link Delegate} that allows buffer handle interop, and
   * such a buffer has been bound to the corresponding input or output {@link Tensor}(s).
   *
   * @param modelId an ID of the target model to run inference
   * @param inputs an array of input data. The inputs should be in the same order as inputs of the
   *     model. Each input can be an array or multidimensional array, or a {@link Buffer} of
   *     primitive types including int, float, long, and byte. {@link Buffer} is the preferred way
   *     to pass large input data, whereas string types require using the (multi-dimensional) array
   *     input path. When {@link Buffer} is used, its content should remain unchanged until model
   *     inference is done, and the caller must ensure that the {@link Buffer} is at the appropriate
   *     read position.
   * @throws IllegalArgumentException if {@code inputs} or {@code outputs} is null or empty, or if
   *     error occurs when running the inference.
   */
  public Tensor[] runForMultipleInputs(
      int modelId, @NonNull Tensor[] inputs) {
    checkNotClosed();
    return wrapper.run(modelId, inputs);
  }

  /**
   * Expicitly updates allocations for all tensors, if necessary.
   *
   * <p>This will propagate shapes and memory allocations for all dependent tensors using the input
   * tensor shape(s) as given.
   *
   * <p>Note: This call is *purely optional*. Tensor allocation will occur automatically during
   * execution if any input tensors have been resized. This call is most useful in determining the
   * shapes for any output tensors before executing the graph, e.g.,
   * <pre>{@code
   * interpreter.resizeInput(0, new int[]{1, 4, 4, 3}));
   * interpreter.allocateTensors();
   * FloatBuffer input = FloatBuffer.allocate(interpreter.getInputTensor(0),numElements());
   * // Populate inputs...
   * FloatBuffer output = FloatBuffer.allocate(interpreter.getOutputTensor(0).numElements());
   * interpreter.run(input, output)
   * // Process outputs...
   * }</pre>
   *
   * @throws IllegalStateException if the graph's tensors could not be successfully allocated.
   */
  public void allocateTensors() {
    checkNotClosed();
    wrapper.allocateTensors();
  }

  /**
   * Resizes idx-th input of the native model to the given dims.
   *
   * @throws IllegalArgumentException if {@code idx} is negtive or is not smaller than the number of
   *     model inputs; or if error occurs when resizing the idx-th input.
   */
  public void resizeInput(int modelId, int idx, @NonNull int[] dims) {
    checkNotClosed();
    wrapper.resizeInput(modelId, idx, dims, false);
  }

  /**
   * Resizes idx-th input of the native model to the given dims.
   *
   * <p>When `strict` is True, only unknown dimensions can be resized. Unknown dimensions are
   * indicated as `-1` in the array returned by `Tensor.shapeSignature()`.
   *
   * @throws IllegalArgumentException if {@code idx} is negtive or is not smaller than the number of
   *     model inputs; or if error occurs when resizing the idx-th input. Additionally, the error
   *     occurs when attempting to resize a tensor with fixed dimensions when `struct` is True.
   */
  public void resizeInput(int modelId, int idx, @NonNull int[] dims, boolean strict) {
    checkNotClosed();
    wrapper.resizeInput(modelId, idx, dims, strict);
  }

  /** Gets the number of input tensors. */
  public int getInputTensorCount(int modelId) {
    checkNotClosed();
    return wrapper.getInputTensorCount(modelId);
  }

  /** Gets the number of output Tensors. */
  public int getOutputTensorCount(int modelId) {
    checkNotClosed();
    return wrapper.getOutputTensorCount(modelId);
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
  public void resetVariableTensors() {
    checkNotClosed();
    wrapper.resetVariableTensors();
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
