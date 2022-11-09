package org.mrsnu.band;

public class NativeWrapper {
  // To simulate the friend class behavior.
  public static final class NativeKey {
    // A key to open the native wrapper in the java object.
    protected NativeKey() {}
  }

  protected static final NativeKey nativeKey = new NativeKey();
  protected long nativeHandle;

  NativeWrapper() {
    nativeHandle = 0;
  }

  protected void checkNotClosed() {
    checkNotClosed("The object has been closed.");
  }

  protected void checkNotClosed(String msg) {
    if (nativeHandle == 0) {
      throw new IllegalStateException(msg);
    }
  }

  public long getNativeHandle() {
    return nativeHandle;
  }
}
