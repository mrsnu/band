#ifndef BAND_BACKEND_FACTORY_H_
#define BAND_BACKEND_FACTORY_H_

#include "band/backend_factory.h"
#include "band/common.h"
#include "band/interface/model.h"
#include "band/interface/model_executor.h"

// Expected workflow (per each backend)
// 1. Implement creators
// 2. Register them with `RegisterBackendCreators` in arbitrary global function
// (e.g., tfl::RegisterCreators)
// 3. Add an arbitrary global function to `RegisterBackendInternal` with
// corresponding flag
// 针对每个后端的预期工作流程如下：
// 1. 实现创建器（creators）。
// 2. 通过 `RegisterBackendCreators` 在任意的全局函数（例如，tfl::RegisterCreators）中注册这些创建器。
// 3. 将某个全局函数以相应的标识符添加至 `RegisterBackendInternal`。

namespace band {

template <typename Base, class... Args>
struct Creator {
 public:
  virtual Base* Create(Args...) const { return nullptr; };
};
// 定义了一个模板结构体，使用c++的模板特化技术，可以实现不同的创建器。
// base是基类，args表示构造base类型的对象可能需要的任意数量和类型的参数
// 定义了一个虚函数Create，返回值是Base*，参数是Args...，默认实现是返回nullptr

class BackendFactory {
    // 设计用来创建和管理不同后端模型执行器、模型和后端工具（Backend Utilities）的工厂类
 public:
  static interface::IModelExecutor* CreateModelExecutor(
      BackendType backend, ModelId model_id, WorkerId worker_id,
      DeviceFlag device_flag,
      CpuSet thread_affinity_mask = BandCPUMaskGetSet(CPUMaskFlag::kAll),
      int num_threads = -1);
    //   创建一个模型执行器(IModelExecutor)实例，根据特定的后端创建模型执行器实例，用于执行模型推理任务。
  static interface::IModel* CreateModel(BackendType backend, ModelId id);
  static interface::IBackendUtil* GetBackendUtil(BackendType backend);
  static std::vector<BackendType> GetAvailableBackends();

  static void RegisterBackendCreators(
    // 为特定的后端注册一组创建器(Creator)对象。
    // 这些创建器负责生成模型执行器、模型和后端工具的实例。
    // 每种后端类型可以有不同的创建器，以支持不同后端的特定实现。
      BackendType backend,
      Creator<interface::IModelExecutor, ModelId, WorkerId, DeviceFlag, CpuSet,
              int>* model_executor_creator,
      Creator<interface::IModel, ModelId>* model_creator,
      Creator<interface::IBackendUtil>* util_creator);

 private:
  BackendFactory() = default;

  static std::map<BackendType,
                  std::shared_ptr<Creator<interface::IModelExecutor, ModelId,
                                          WorkerId, DeviceFlag, CpuSet, int>>>
      model_executor_creators_;
    //   存储每种后端类型对应的模型执行器创建器。这是一个映射，将后端类型映射到创建模型执行器的创建器对象。
  static std::map<BackendType,
                  std::shared_ptr<Creator<interface::IModel, ModelId>>>
      model_creators_;
    //    存储每种后端类型对应的模型创建器
  static std::map<BackendType,
                  std::shared_ptr<Creator<interface::IBackendUtil>>>
      util_creators_;
    //   存储每种后端类型对应的后端工具创建器
};
}  // namespace band

#endif