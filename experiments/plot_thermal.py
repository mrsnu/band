import json
import matplotlib.pyplot as plt

def plot_thermal(log_path):
  with open(log_path, 'r') as f:
    data = json.load(f)['traceEvents']
  expected_cpu = [d["args"]["expected_therm"]["CPU"] for d in data if d["ph"] == "E"]
  profiled_cpu = [d["args"]["profiled_therm"]["CPU"] for d in data if d["ph"] == "E"]
  expected_gpu = [d["args"]["expected_therm"]["GPU"] for d in data if d["ph"] == "E"]
  profiled_gpu = [d["args"]["profiled_therm"]["GPU"] for d in data if d["ph"] == "E"]
  expected_dsp = [d["args"]["expected_therm"]["DSP"] for d in data if d["ph"] == "E"]
  profiled_dsp = [d["args"]["profiled_therm"]["DSP"] for d in data if d["ph"] == "E"]
  expected_npu = [d["args"]["expected_therm"]["NPU"] for d in data if d["ph"] == "E"]
  profiled_npu = [d["args"]["profiled_therm"]["NPU"] for d in data if d["ph"] == "E"]
  expected_target = [d["args"]["expected_therm"]["Target"] for d in data if d["ph"] == "E"]
  profiled_target = [d["args"]["profiled_therm"]["Target"] for d in data if d["ph"] == "E"]
  
  cpu_diff = [abs(e - p) for e, p in zip(expected_cpu, profiled_cpu)]
  gpu_diff = [abs(e - p) for e, p in zip(expected_gpu, profiled_gpu)]
  dsp_diff = [abs(e - p) for e, p in zip(expected_dsp, profiled_dsp)]
  npu_diff = [abs(e - p) for e, p in zip(expected_npu, profiled_npu)]
  target_diff = [abs(e - p) for e, p in zip(expected_target, profiled_target)]
  
  cpu_rmse = (sum([d**2 for d in cpu_diff]) / len(cpu_diff))**0.5
  gpu_rmse = (sum([d**2 for d in gpu_diff]) / len(gpu_diff))**0.5
  dsp_rmse = (sum([d**2 for d in dsp_diff]) / len(dsp_diff))**0.5
  npu_rmse = (sum([d**2 for d in npu_diff]) / len(npu_diff))**0.5
  target_rmse = (sum([d**2 for d in target_diff]) / len(target_diff))**0.5
  
  print("CPU RMSE: ", cpu_rmse)
  print("GPU RMSE: ", gpu_rmse)
  print("DSP RMSE: ", dsp_rmse)
  print("NPU RMSE: ", npu_rmse)
  print("Target RMSE: ", target_rmse)
  
  # Plot All in subplots
  fig, axs = plt.subplots(5, 1, figsize=(10, 10))
  axs[0].plot(expected_cpu, label="expected")
  axs[0].plot(profiled_cpu, label="profiled")
  axs[0].set_title("CPU")
  axs[0].legend()
  axs[1].plot(expected_gpu, label="expected")
  axs[1].plot(profiled_gpu, label="profiled")
  axs[1].set_title("GPU")
  axs[1].legend()
  axs[2].plot(expected_dsp, label="expected")
  axs[2].plot(profiled_dsp, label="profiled")
  axs[2].set_title("DSP")
  axs[2].legend()
  axs[3].plot(expected_npu, label="expected")
  axs[3].plot(profiled_npu, label="profiled")
  axs[3].set_title("NPU")
  axs[3].legend()
  axs[4].plot(expected_target, label="expected")
  axs[4].plot(profiled_target, label="profiled")
  axs[4].set_title("Target")
  axs[4].legend()
  plt.tight_layout()
  
  plt.savefig("thermal.png")

if __name__  == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--log", type=str, default="log.json")
  args = parser.parse_args()
  plot_thermal(args.log)