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

import java.util.logging.Logger;

/** Static utility methods for loading the Band runtime and native code. */
public final class Band {
  private static final Logger logger = Logger.getLogger(Band.class.getName());
  private static final String BAND_RUNTIME_LIBNAME = "band_jni";

  private static final Throwable LOAD_LIBRARY_EXCEPTION;
  private static volatile boolean isInit = false;

  static {
    Throwable loadLibraryException = null;
    try {
      System.loadLibrary(BAND_RUNTIME_LIBNAME);
    } catch (UnsatisfiedLinkError e) {
      if (loadLibraryException == null) {
        loadLibraryException = e;
      } else {
        loadLibraryException.addSuppressed(e);
      }
    }
    LOAD_LIBRARY_EXCEPTION = loadLibraryException;
  }

  private Band() {
  }

  public static void init() {
    if (isInit) {
      return;
    }

    try {
      nativeDoNothing();
      isInit = true;
    } catch (UnsatisfiedLinkError e) {
      Throwable exceptionToLog = LOAD_LIBRARY_EXCEPTION != null ? LOAD_LIBRARY_EXCEPTION : e;
      UnsatisfiedLinkError exceptionToThrow = new UnsatisfiedLinkError(
          "Failed to load native Band methods. Check that the correct native"
              + " libraries are present, and, if using a custom native library, have been"
              + " properly loaded via System.loadLibrary():\n"
              + "  "
              + exceptionToLog);
      exceptionToThrow.initCause(e);
      throw exceptionToThrow;
    }
  }

  public static void setVerbosity(LogSeverity severity) {
    nativeSetVerbosity(severity.getValue());
  }

  public static String getLastLog() {
    return nativeGetLastLog();
  }

  private static native void nativeSetVerbosity(int severity);
  private static native String nativeGetLastLog();
  private static native void nativeDoNothing();

}