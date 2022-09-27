#include "band/scheduler/fixed_device_scheduler.h"

#include "band/error_reporter.h"

namespace Band {
ScheduleAction FixedDeviceScheduler::Schedule(const Context& context,
                                              JobQueue& requests) {
  ScheduleAction action;

  // TODO: fallback subgraphs for FixedDevicePlanner?
  while (!requests.empty()) {
    Job to_execute = requests.front();
    requests.pop_front();

    int model_id = to_execute.model_id;
    int worker_id = to_execute.subgraph_key.GetWorkerId();

    BAND_NOT_IMPLEMENTED;

    // if (0 <= to_execute.device_id && to_execute.device_id < kBandNumDevices)
    // {
    //   BandDeviceFlags device_flag =
    //       static_cast<BandDeviceFlags>(to_execute.device_id);
    //   // TODO: Select any available device worker
    //   BAND_REPORT_ERROR(DefaultErrorReporter(), "NOT IMPLEMENTED");
    //   continue;
    // } else {
    //   // TODO: Support model assignment for fixed device scheduler
    //   // worker_id = planner_->GetModelWorkerMap()[model_id];
    //   BAND_REPORT_ERROR(DefaultErrorReporter(), "NOT IMPLEMENTED");
    //   continue;
    // }

    SubgraphKey key = context.GetModelSubgraphKey(model_id, worker_id);
    action[worker_id].push_back({to_execute, key});
  }
  return action;
}

}  // namespace Band
