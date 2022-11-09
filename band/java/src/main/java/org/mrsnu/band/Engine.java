package org.mrsnu.band;

import java.util.List;

public class Engine implements BandInterface, AutoCloseable {
  private NativeEngineWrapper wrapper;

  public Engine(Config config) {
    wrapper = new NativeEngineWrapper(config);
  }

  protected void checkNotClosed() {
    if (wrapper == null) {
      throw new IllegalStateException("The engine has been closed.");
    }
  }

  public void registerModel(Model model) {
    checkNotClosed();
    wrapper.registerModel(model);
  }

  public int getNumInputTensors(Model model) {
    checkNotClosed();
    return wrapper.getNumInputTensors(model);
  }

  public int getNumOutputTensors(Model model) {
    checkNotClosed();
    return wrapper.getNumOutputTensors(model);
  }

  public void requestSync(Model model, List<Tensor> inputTensors, List<Tensor> outputTensors) {
    checkNotClosed();

    wrapper.requestSync(model, inputTensors, outputTensors);
  }

  public void requestAsync(Model model, List<Tensor> inputTensors) {
    checkNotClosed();
    wrapper.requestAsync(model, inputTensors);
  }

  public void wait(Request request, List<Tensor> outputTensors) {
    checkNotClosed();
    wrapper.wait(request, outputTensors);
  }

  public Tensor createInputTensor(Model model, int index) {
    checkNotClosed();
    wrapper.createInputTensor(model, index);
    return null;
  }

  public Tensor createOutputTensor(Model model, int index) {
    checkNotClosed();
    wrapper.createOutputTensor(model, index);
    return null;
  }

  @Override
  public void close() {
    if (wrapper != null) {
      wrapper.close();
      wrapper = null;
    }
  }

  public NativeEngineWrapper getNativeWrapper(NativeWrapper.NativeKey key) {
    return wrapper;
  }
}
