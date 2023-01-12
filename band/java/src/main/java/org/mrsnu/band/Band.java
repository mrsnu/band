package org.mrsnu.band;

import java.util.logging.Logger;

/** Static utility methods for loading the Band runtime and native code. */
public final class Band {
  private static final Logger logger = Logger.getLogger("mylogger");
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

  private Band() {}
  
  public static void init() {
    if (isInit) {
      return;
    }

    try {
      nativeDoNothing();
      isInit = true;
    } catch (UnsatisfiedLinkError e) {
      Throwable exceptionToLog = LOAD_LIBRARY_EXCEPTION != null ? LOAD_LIBRARY_EXCEPTION : e;
      UnsatisfiedLinkError exceptionToThrow =
          new UnsatisfiedLinkError(
              "Failed to load native Band methods. Check that the correct native"
                  + " libraries are present, and, if using a custom native library, have been"
                  + " properly loaded via System.loadLibrary():\n"
                  + "  "
                  + exceptionToLog);
      exceptionToThrow.initCause(e);
      throw exceptionToThrow;
    }
  }

  private static native void nativeDoNothing();
  
}
