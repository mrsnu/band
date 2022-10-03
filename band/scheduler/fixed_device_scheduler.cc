#include "band/scheduler/fixed_device_scheduler.h"

#include "band/error_reporter.h"

namespace Band {
ScheduleAction FixedDeviceScheduler::Schedule(const Context& context,
                                              JobQueue& requests) {
  ScheduleAction action;

  // TODO: fallback subgraphs for FixedDevicePlanner?
  while (!requests.empty()) { 
    Job to_execute = requests.front();
    requests.pop_front(); // erase job

    int model_id = to_execute.model_id;
    ModelConfig model_config = context.GetModelConfig(model_id);
    int device_id = model_config.device;

    if (0 <= device_id && device_id < kBandNumDevices)
    {
      WorkerId worker_id = context.GetModelWorker(model_id);
      SubgraphKey key = context.GetModelSubgraphKey(model_id, worker_id);
      action[worker_id].push_back({to_execute, key});
    } else {
      BAND_REPORT_ERROR(DefaultErrorReporter(), "Invalid device for model %s", 
          model_config.model_fname);
    }
  }
  return action;
}

}  // namespace Band
