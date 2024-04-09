#ifndef BAND_CONFIG_H_
#define BAND_CONFIG_H_

#include <limits>
#include <string>
#include <vector>

#include "band/common.h"

namespace band {

struct ProfileConfig {
  ProfileConfig() {}
  bool online = true;
  // 是否在运行时进行性能分析
  int num_warmups = 1;
  // 运行时进行性能分析的预热次数
  int num_runs = 1;
  // 运行时进行性能分析的运行次数
  std::string profile_data_path = "";
  // 存储性能分析数据的路径
  float smoothing_factor = 0.1;
  // 平滑因子
};

struct PlannerConfig {
  int schedule_window_size = std::numeric_limits<int>::max();
  // 调度窗口大小
  std::vector<SchedulerType> schedulers;
  // 调度器类型
  CPUMaskFlag cpu_mask = CPUMaskFlag::kAll;
  // CPU掩码
  std::string log_path = "";
  // 日志路径
};

struct WorkerConfig {
  WorkerConfig() {
    // Add one default worker per device
    for (size_t i = 0; i < EnumLength<DeviceFlag>(); i++) {
      workers.push_back(static_cast<DeviceFlag>(i));
      // 将所有设备添加到workers中
    }
    cpu_masks =
        std::vector<CPUMaskFlag>(EnumLength<DeviceFlag>(), CPUMaskFlag::kAll);
    num_threads = std::vector<int>(EnumLength<DeviceFlag>(), 1);
  }
  std::vector<DeviceFlag> workers;
  std::vector<CPUMaskFlag> cpu_masks;
  std::vector<int> num_threads;
  bool allow_worksteal = false;
  // 是否允许工作窃取
  int availability_check_interval_ms = 30000;
  // 可用性检查间隔
};

struct SubgraphConfig {
  int minimum_subgraph_size = 7;
  // 最小子图大小
  SubgraphPreparationType subgraph_preparation_type =
      SubgraphPreparationType::kMergeUnitSubgraph;
  // 子图准备类型
};

struct ResourceMonitorConfig {
  std::string log_path = "";
  // 日志路径
  std::map<DeviceFlag, std::string> device_freq_paths;
  // 设备频率路径
  int monitor_interval_ms = 10;
};

struct RuntimeConfig {
  // 聚合了所有的配置
  CPUMaskFlag cpu_mask;
  SubgraphConfig subgraph_config;
  ProfileConfig profile_config;
  PlannerConfig planner_config;
  WorkerConfig worker_config;
  ResourceMonitorConfig resource_monitor_config;

 private:
  friend class RuntimeConfigBuilder;
  // 暗示配置可能通过构建器模式进行设置。
  RuntimeConfig() { cpu_mask = CPUMaskFlag::kAll; };
};

}  // namespace band
#endif  // BAND_CONFIG_H_
