/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.mrsnu.band;

import java.util.List;

public class Engine {
  private NativeEngineWrapper wrapper;

  public Engine() {
    Band.init();
    wrapper = new NativeEngineWrapper();
  }

  public Engine(Config config) {
    Band.init();
    wrapper = new NativeEngineWrapper(config);
  }

  public void registerModel(Model model) {
    wrapper.registerModel(model);
  }

  public int getNumInputTensors(Model model) {
    return wrapper.getNumInputTensors(model);
  }

  public int getNumOutputTensors(Model model) {
    return wrapper.getNumOutputTensors(model);
  }

  public void requestSync(Model model, List<Tensor> inputTensors, List<Tensor> outputTensors) {
    wrapper.requestSync(model, inputTensors, outputTensors);
  }

  public Request requestAsync(Model model, List<Tensor> inputTensors) {
    return wrapper.requestAsync(model, inputTensors);
  }

  public List<Request> requestAsyncBatch(List<Model> models, List<List<Tensor>> inputTensorLists) {
    return wrapper.requestAsyncBatch(models, inputTensorLists);
  }

  public void wait(Request request, List<Tensor> outputTensors) {
    wrapper.wait(request, outputTensors);
  }

  public Tensor createInputTensor(Model model, int index) {
    return wrapper.createInputTensor(model, index);
  }

  public Tensor createOutputTensor(Model model, int index) {
    return wrapper.createOutputTensor(model, index);
  }
}
