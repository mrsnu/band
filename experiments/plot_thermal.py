import json
import matplotlib.pyplot as plt

def plot_thermal(log_path, expected=True):
  with open(log_path, 'r') as f:
    data = json.load(f)
  
  cpu = {
    "total": [],
  }
  gpu = {
    "total": [],
  }
  dsp = {
    "total": [],
  }
  npu = {
    "total": [],
  }
  target = {
    "total": [],
  }
  for d in data:
    if d["expected"] != expected:
      continue
    if d["key"] not in cpu or d["key"] not in gpu or d["key"] not in dsp or d["key"] not in npu or d["key"] not in target:
      cpu[d["key"]] = []
      gpu[d["key"]] = []
      dsp[d["key"]] = []
      npu[d["key"]] = []
      target[d["key"]] = []
    cpu[d["key"]].append(d["therm_end"]["CPU"])
    gpu[d["key"]].append(d["therm_end"]["GPU"])
    dsp[d["key"]].append(d["therm_end"]["DSP"])
    npu[d["key"]].append(d["therm_end"]["NPU"])
    target[d["key"]].append(d["therm_end"]["Target"])
    
    # total
    cpu["total"].append(d["therm_end"]["CPU"])
    gpu["total"].append(d["therm_end"]["GPU"])
    dsp["total"].append(d["therm_end"]["DSP"])
    npu["total"].append(d["therm_end"]["NPU"])
    target["total"].append(d["therm_end"]["Target"])
    
  for key in cpu:
    plt.plot(cpu[key], label="CPU")
    plt.plot(gpu[key], label="GPU")
    plt.plot(dsp[key], label="DSP")
    plt.plot(npu[key], label="NPU")
    plt.plot(target[key], label="Target")
    plt.xlabel("Time (us)")
    plt.ylabel("Temperature (C)")
    # Bound 20-80
    plt.ylim(20, 80)
    plt.legend()
    plt.savefig(f"thermal_{key}_{expected}.png")
    plt.clf()
    
  plt.plot(cpu["total"], label="CPU")
  plt.plot(gpu["total"], label="GPU")
  plt.plot(dsp["total"], label="DSP")
  plt.plot(npu["total"], label="NPU")
  plt.plot(target["total"], label="Target")
  plt.xlabel("Time (us)")
  plt.ylabel("Temperature (C)")
  # Bound 20-80
  plt.ylim(20, 80)
  plt.legend()
  plt.savefig(f"thermal_total_{expected}.png")
  plt.clf()
  
if __name__  == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--log", type=str, default="thermal.json")
  parser.add_argument("--expected", action="store_true")
  args = parser.parse_args()
  plot_thermal(args.log, args.expected)