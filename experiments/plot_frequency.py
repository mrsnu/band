import json
import matplotlib.pyplot as plt

def plot_frequency(log_path):
  with open(log_path, 'r') as f:
    data = json.load(f)

  time = []
  cpu = []
  gpu = []
  dsp = []
  npu = []
  target = []
  for d in data:
    time.append(d["time"])
    cpu.append(d["frequency"]["CPU"])
    gpu.append(d["frequency"]["GPU"])
  start_time = time[0]
  for i in range(len(time)):
    time[i] = time[i] - start_time
  plt.plot(time, cpu, label="CPU")
  plt.plot(time, gpu, label="GPU")
  plt.xlabel("Time (us)")
  plt.ylabel("Frequency (GHz)")
  plt.legend()
  plt.savefig("frequency.png")
  
if __name__  == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--log", type=str, default="frequency.json")
  args = parser.parse_args()
  plot_frequency(args.log)