#include <stdarg.h>
#include <stdint.h>

#include <array>
#include <fstream>
#include <vector>

#include "band/backend/tfl/model.h"
#include "band/backend/tfl/model_executor.h"
#include "band/backend/tfl/tensor.h"
#include "band/backend_factory.h"
#include "band/buffer/buffer.h"
#include "band/buffer/common_operator.h"
#include "band/buffer/image_operator.h"
#include "band/buffer/image_processor.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/interface/model_executor.h"
#include "band/interface/tensor.h"
#include "band/model.h"
#include "band/tensor.h"
#include "band/test/image_util.h"

namespace band {
namespace test {

void test() {
  RuntimeConfigBuilder b;
  RuntimeConfig config =
      b.AddPlannerLogPath("band/test/data/log.json")
          .AddSchedulers({SchedulerType::kHeterogeneousEarliestFinishTime})
          .AddMinimumSubgraphSize(7)
          .AddSubgraphPreparationType(
              SubgraphPreparationType::kUnitSubgraph)
          .AddCPUMask(CPUMaskFlag::kAll)
          .AddPlannerCPUMask(CPUMaskFlag::kPrimary)
#ifdef __ANDROID__
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kDSP, DeviceFlag::kNPU,
                       DeviceFlag::kGPU})
          .AddWorkerNumThreads({3, 1, 1, 1})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kAll,
                              CPUMaskFlag::kAll, CPUMaskFlag::kAll})
          .AddThermalWindowSize(1000)
          .AddThermLogPath("band/test/data/thermal.log")
          .AddFreqLogPath("band/test/data/freq.log")
          .AddCPUFreqPath("/sys/devices/system/cpu/cpu7/cpufreq")
          .AddGPUFreqPath("/sys/class/devfreq/2c00000.qcom,kgsl-3d0")
          .AddCPUThermIndex(6)
          .AddGPUThermIndex(32)
          .AddNPUThermIndex(24)
          .AddDSPThermIndex(20)
          .AddTargetThermIndex(75)
#else
          .AddWorkers({DeviceFlag::kCPU, DeviceFlag::kCPU})
          .AddWorkerNumThreads({3, 4})
          .AddWorkerCPUMasks({CPUMaskFlag::kBig, CPUMaskFlag::kLittle})
#endif  // __ANDROID__
          .AddLatencySmoothingFactor(0.1)
          .AddProfilePath("band/test/data/profile.json")
          .AddNumWarmups(1)
          .AddNumRuns(1)
          .AddAvailabilityCheckIntervalMs(30000)
          .AddScheduleWindowSize(10)
          .Build();

  auto engine = Engine::Create(config);

  Model model;
  {
    auto status = model.FromPath(BackendType::kTfLite,
                                 "band/test/data/retinaface-mbv2-int8.tflite");
    if (!status.ok()) {
      std::cout << status.message() << std::endl;
    }
  }
  {
    auto status = engine->RegisterModel(&model);
    if (!status.ok()) {
      std::cout << status.message() << std::endl;
    }
  }

  Model model2;

  {
    auto status = model2.FromPath(BackendType::kTfLite,
                                  "band/test/data/arc-mbv2-int8.tflite");
    if (!status.ok()) {
      std::cout << status.message() << std::endl;
    }
  }
  {
    auto status = engine->RegisterModel(&model2);
    if (!status.ok()) {
      std::cout << status.message() << std::endl;
    }
  }

  for (size_t worker_id = 0; worker_id < engine->GetNumWorkers(); worker_id++) {
    {
      auto status = engine->RequestSync(
          model.GetId(), {static_cast<int>(worker_id), true, -1, -1});
      if (!status.ok()) {
        std::cout << status.message() << std::endl;
      }
    }

    {
      auto status = engine->RequestSync(
          model.GetId(), {static_cast<int>(worker_id), false, -1, -1});
      if (!status.ok()) {
        std::cout << status.message() << std::endl;
      }
    }
  }
}

}  // namespace test
}  // namespace band

int main() {
  band::test::test();
  return 0;
}