#include "tensorflow/lite/planner/random_assign_scheduler.h"
#include <random>
#include "tensorflow/lite/profiling/time.h"
#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#else
#define LOGI(...) printf(__VA_ARGS__)
#endif // defined(__ANDROID__)
namespace tflite {
namespace impl {

void RandomAssignScheduler::Schedule(JobQueue& requests) {
  // LOGI("Idle worker size : %d", idle_workers.size());
  while (!requests.empty()) {
    std::set<int> idle_workers = planner_->GetIdleAllWorkers();
    // Select a worker
    int target_idx = rand() % idle_workers.size();
    std::set<int>::iterator it = idle_workers.begin();
    std::advance(it, target_idx);
    int worker_id = *it;
    // LOGI("It's selected : %d", worker_id);
    // int worker_id = kTfLiteCLOUD;

    Job to_execute = requests.front();
    requests.pop_front();
    int model_id = to_execute.model_id;

    // Get a subgraph to execute
    int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, worker_id);
    Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);

    EnqueueAction(to_execute, subgraph);
  }
}

}  // namespace impl
}  // namespace tflite
