package org.mrsnu.band;

public class RequestOption {
  private int target_worker = -1;
  private boolean require_callback = true;
  private int slo_us = -1;
  private float slo_scale = -1.f;

  public RequestOption() {}

  public RequestOption(int target_worker, boolean require_callback, int slo_us, float slo_scale) {
    this.target_worker = target_worker;
    this.require_callback = require_callback;
    this.slo_us = slo_us;
    this.slo_scale = slo_scale;
  }

  public int getTargetWorker() {
    return target_worker;
  }

  public boolean getRequireCallback() {
    return require_callback;
  }

  public int getSloUs() {
    return slo_us;
  }

  public float getSloScale() {
    return slo_scale;
  }
}