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

import java.nio.ByteBuffer;
import java.util.List;

public class Model {
  private NativeModelWrapper wrapper;
  
  public Model(BackendType backendType, String filePath) {
    Band.init();
    wrapper = new NativeModelWrapper();
    wrapper.loadFromFile(backendType, filePath);
  }
  
  public Model(BackendType backendType, ByteBuffer modelBuffer) {
    Band.init();
    wrapper = new NativeModelWrapper();
    wrapper.loadFromBuffer(backendType, modelBuffer);
  }

  public List<BackendType> getSupportedBackends() {
    return wrapper.getSupportedBackends();
  }

  public long getNativeHandle() {
    return wrapper.getNativeHandle();
  }
}
