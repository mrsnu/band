import json
import matplotlib.pyplot as plt

def plot_latency(log_path):
  with open(log_path, "r") as f:
    data = json.load(f)["traceEvents"]
  
  """
  data = {
    "name": str,
    "ts": int,
    "args": {
      "expected_latency": float,
      "profiled_latency": float
    }
  }
  """
  expected_latency = [d["args"]["expected_latency"] for d in data if d["ph"] == "E"]
  profiled_latency = [d["args"]["profiled_latency"] for d in data if d["ph"] == "E"]
  # Sort with 'ts'
  expected_latency = [e for _, e in sorted(zip([d["ts"] for d in data if d["ph"] == "E"], expected_latency))]
  profiled_latency = [p for _, p in sorted(zip([d["ts"] for d in data if d["ph"] == "E"], profiled_latency))]
  
  latency_diff = [abs(e - p) for e, p in zip(expected_latency, profiled_latency)]
  latency_rmse = (sum([d**2 for d in latency_diff]) / len(latency_diff))**0.5
  
  print("Latency RMSE: ", latency_rmse)
  
  plt.plot(expected_latency, label="expected")
  plt.plot(profiled_latency, label="profiled")
  plt.title("Latency")
  plt.legend()
  plt.tight_layout()
  
  plt.savefig("latency.png")
  

if __name__  == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--log", type=str, default="log.json")
  args = parser.parse_args()
  plot_latency(args.log)