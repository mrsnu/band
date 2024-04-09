#include "band/model_analyzer.h"

#include <algorithm>
#include <iterator>
#include <memory>

#include "absl/strings/str_format.h"
#include "band/backend_factory.h"
#include "band/engine_interface.h"
#include "band/interface/model.h"
#include "band/interface/model_executor.h"
#include "band/logger.h"
#include "band/model.h"
#include "band/worker.h"

namespace band {
/**
 * Converts a set of integers to a string representation.
 *
 * @param set The set of integers to convert.
 * @return The string representation of the set.
 */
std::string SetToString(const std::set<int>& set) {
  auto range_to_string = [](int lhs, int rhs) {
    if (lhs == rhs) {
      return std::to_string(lhs);
    } else {
      return std::to_string(lhs) + "-" + std::to_string(rhs);
    }
  };

  std::string result = "{";
  if (set.size() > 0) {
    int current_start = std::numeric_limits<int>::min();
    // 用于追踪当前连续范围内的起始值
    int prev = current_start;
    for (auto v : set) {
      // not continuous
      if (v > prev + 1) {
        // 如果v不连续
        if (current_start >= 0) {
          result += range_to_string(current_start, prev) + ",";
        }
        current_start = v;
      }
      prev = v;
    }
    result += range_to_string(current_start, *set.rbegin());
  }
  return result + "}";
}

/**
 * Converts the SubgraphDef object to a string representation.
 *将一个子图的定义转换成一个字符串形式的描述，便于调试或日志记录。
 * @return A string representation of the SubgraphDef object.
 */
std::string SubgraphDef::ToString() const {
  return "Index " + SetToString(unit_subgraph_indices) + " Ops " +
         SetToString(op_indices);
        //  单元子图索引集合 (unit_subgraph_indices) 和操作索引集合 (op_indices) 的字符串表示
}

/**
 * @brief The std::string class represents a sequence of characters.
 * 
 * It provides various member functions to manipulate and access the characters in the string.
 * 为一系列子图定义（由 std::vector<SubgraphDef> 表示）生成一个摘要，
 * 展示哪些子图是独立的单元，哪些是合并的子图，并显示它们在不同工作器上的可用性。
 * @note This class is part of the C++ Standard Library.
 */
std::string SummarizeSubgraphs(const std::vector<SubgraphDef>& subgraph_defs) {
  std::string summary = "\n";
  std::vector<SubgraphDef> unit_subgraphs;
  // 用于存放只包含一个单元子图索引的子图定义
  std::vector<SubgraphDef> merged_subgraphs;
  // 用于存放包含多个单元子图索引的子图定义
  std::set<int> unique_unit_subgraph_indices;
  // 存储所有独立单元子图索引的集合，用于后续统计和显示
  int num_workers = 0;
  // 记录工作器的数量

  for (const auto& subgraph_def : subgraph_defs) {
    // 遍历所有子图定义
    if (subgraph_def.unit_subgraph_indices.size() == 1) {
      // 将其视为独立单元子图
      unit_subgraphs.push_back(subgraph_def);
      // 将其添加到独立单元子图集合中
      unique_unit_subgraph_indices.insert(
          *subgraph_def.unit_subgraph_indices.begin());
          // 将其单元子图索引添加到集合中
    } else {
      merged_subgraphs.push_back(subgraph_def);
      // >1 视其为合并子图 加入合并子图集合
    }
    num_workers = std::max(num_workers, subgraph_def.worker_id + 1);
  }

//处理单元独立子图
  if (unit_subgraphs.size()) {
    // 如果存在单元独立子图
    summary += "UnitSubgraph Definitions\n";

    std::map<WorkerId, std::vector<bool>> unit_subgraph_availabilities;
    // 初始化映射 为每一个工作器创建一个布尔值向量 表示每个独立单元子图的可用性
    for (WorkerId i = 0; i < num_workers; i++) {
      // 遍历所有工作器
      unit_subgraph_availabilities[i] =
          std::vector<bool>(unique_unit_subgraph_indices.size(), false);
          // 为每个工作器创建一个布尔值向量，长度为独立单元子图索引集合的大小，初始值为false
    }

    for (const auto& unit_subgraph : unit_subgraphs) {
      unit_subgraph_availabilities[unit_subgraph.worker_id]
                                  [*unit_subgraph.unit_subgraph_indices
                                        .begin()] = true;
                                        // 设置对应工作器和子图索引的可用性为true
      if (unit_subgraph.worker_id == 0) {
        summary += "\t" + unit_subgraph.ToString() + "\n";
      }
    }

    summary += "UnitSubgraph Availabilities\n";

    for (const auto& unit_subgraph_availability :
         unit_subgraph_availabilities) {
          // 显示独立单元子图的可用性
      summary += "\t Worker " +
                 std::to_string(unit_subgraph_availability.first) + "\t";
      for (const auto& availability : unit_subgraph_availability.second) {
        summary += (availability ? "O\t" : "X\t");
      }
      summary += "\n";
    }
  }

// 处理包含多个子图索引的合并子图
  if (merged_subgraphs.size()) {
    summary += "MergedSubgraphs\n";

    for (WorkerId target_worker_id = 0; target_worker_id < num_workers;
         target_worker_id++) {
          // 遍历所有工作器
      for (const auto& merged_subgraph : merged_subgraphs) {
        // 内层循环遍历 merged_subgraphs 集合中的每一个子图 merged_subgraph
        if (merged_subgraph.worker_id == target_worker_id) {
          // 当前合并子图的 worker_id 与外层循环的 target_worker_id 相匹配，说明这个子图是由该工作器处理的
          summary += "\t Worker " + std::to_string(target_worker_id) + "\t";
          for (const auto& unit_index : unique_unit_subgraph_indices) {
            summary +=
                (merged_subgraph.unit_subgraph_indices.find(unit_index) !=
                         merged_subgraph.unit_subgraph_indices.end()
                     ? "-\t"
                     : " \t");
          }
          summary += "\n";
        }
      }
    }
  }

  return summary;
}

/**
 * @brief 
 * This class provides a convenient way to manipulate strings of characters.
 * It supports various operations such as concatenation, substring extraction,
 * and searching. It is part of the Standard Library and is defined in the
 * `std` namespace.
 */
std::string SummarizeFallbackPerWorkerSubgraphs(
    const std::vector<SubgraphDef>& unit_subgraph_defs,
    const std::vector<SubgraphDef>& subgraph_defs) {
  std::string summary = SummarizeSubgraphs(unit_subgraph_defs);

  std::set<int> unique_unit_subgraph_indices;
  int num_workers = 0;
  for (const auto& subgraph_def : unit_subgraph_defs) {
    // 确定涉及的工作器数量和单元子图索引集合
    if (subgraph_def.unit_subgraph_indices.size() == 1) {
      unique_unit_subgraph_indices.insert(
          *subgraph_def.unit_subgraph_indices.begin());
    }
    num_workers = std::max(num_workers, subgraph_def.worker_id + 1);
  }

  summary += "FallbackPerWorkerSubgraphs\n";

  for (WorkerId target_worker_id = 0; target_worker_id < num_workers;
       target_worker_id++) {
    for (const auto& merged_subgraph : subgraph_defs) {
      if (merged_subgraph.worker_id == target_worker_id) {
        summary += "\t Worker " + std::to_string(target_worker_id) + "\t";
        for (const auto& unit_index : unique_unit_subgraph_indices) {
          summary += (merged_subgraph.unit_subgraph_indices.find(unit_index) !=
                              merged_subgraph.unit_subgraph_indices.end()
                          ? "-\t"
                          : " \t");
        }
        summary += "\n";
      }
    }
  }

  return summary;
}

/**
 * @brief Constructs a ModelAnalyzer object.
 *
 * This constructor initializes a ModelAnalyzer object with the provided parameters.
 *
 * @param engine The reference to the IEngine object.
 * @param need_fallback_subgraph A boolean indicating whether a fallback subgraph is needed.
 * @param subgraph_config The SubgraphConfig object specifying the subgraph configuration.
 * @param model A pointer to the Model object.
 * @param backend_type The BackendType specifying the type of the backend.
 */
ModelAnalyzer::ModelAnalyzer(const IEngine& engine, bool need_fallback_subgraph,
                             SubgraphConfig subgraph_config, Model* model,
                             BackendType backend_type)
    : engine_(engine),
      need_fallback_subgraph_(need_fallback_subgraph),
      subgraph_config_(subgraph_config),
      backend_type_(backend_type) {
  std::unique_ptr<interface::IModelExecutor> interpreter(
      BackendFactory::CreateModelExecutor(backend_type, model->GetId(), 0,
                                          DeviceFlag::kCPU));
  // TODO(widiba03304): Report error when it fails.
  model_spec_ = std::make_shared<ModelSpec>(
      interpreter->InvestigateModelSpec(model->GetBackendModel(backend_type))
          .value());
          // model->GetBackendModel(backend_type) 从模型中获取针对特定后端类型的模型数据
  // nterpreter->InvestigateModelSpec(...).value() 调用模型执行器的 InvestigateModelSpec 方法，
  // 该方法分析模型的规范并返回相关的信息（如模型可以运行的操作和不支持的设备）

  for (auto device_unsupported_ops : model_spec_->unsupported_ops) {
    BAND_LOG_DEBUG("Unsupported ops %s (%s)",
                   SetToString(device_unsupported_ops.second).c_str(),
                   ToString(device_unsupported_ops.first));
  }

  for (auto device : model_spec_->unavailable_devices) {
    BAND_LOG_DEBUG("Unsupported devices %s", ToString(device));
  }
}

/**
 * @brief Creates subgraphs for the model analyzer.
 * 
 * This function creates subgraphs for the model analyzer based on the specified subgraph preparation type.
 * 
 * @return absl::StatusOr<std::pair<ModelSpec, std::vector<SubgraphDef>>> - A status or a pair containing the model specification and the vector of subgraph definitions.
 */
absl::StatusOr<std::pair<ModelSpec, std::vector<SubgraphDef>>>
ModelAnalyzer::CreateSubgraphs() {
  std::vector<SubgraphDef> subgraph_defs;
  std::vector<SubgraphDef> unit_subgraph_defs;

  // TODO(widiba03304): Add error propagation logic.
  auto status = GetUnitSubgraphs(unit_subgraph_defs);
  // 获取单元子图
  if (!status.ok()) {
    return status;
  }

  switch (subgraph_config_.subgraph_preparation_type) {
    case SubgraphPreparationType::kFallbackPerWorker: {
      // 对每个工作器，收集特定于回退操作的子图，并在这些工作器特定子图中合并单元子图索引。
      for (WorkerId worker_id = 0; worker_id < engine_.GetNumWorkers();
           worker_id++) {
        std::vector<SubgraphDef> worker_subgraphs =
            GetSubgraphsForFallbackOps(worker_id);

        for (SubgraphDef& worker_subgraph : worker_subgraphs) {
          // set unit subgraph indices
          for (int unit_subgraph_id = 0;
               unit_subgraph_id < unit_subgraph_defs.size();
               unit_subgraph_id++) {
            // add all unit subgraphs that are part of the worker subgraph
            if (std::includes(
                    worker_subgraph.op_indices.begin(),
                    worker_subgraph.op_indices.end(),
                    unit_subgraph_defs[unit_subgraph_id].op_indices.begin(),
                    unit_subgraph_defs[unit_subgraph_id].op_indices.end())) {
              worker_subgraph.unit_subgraph_indices.insert(
                  unit_subgraph_defs[unit_subgraph_id]
                      .unit_subgraph_indices.begin(),
                  unit_subgraph_defs[unit_subgraph_id]
                      .unit_subgraph_indices.end());
            }
            // 将属于工作线程子图的单元子图的索引添加到工作线程子图的 unit_subgraph_indices中
          }
        }

        subgraph_defs.insert(subgraph_defs.end(), worker_subgraphs.begin(),
                             worker_subgraphs.end());
      }
    } break;
    case SubgraphPreparationType::kNoFallbackSubgraph:
    case SubgraphPreparationType::kUnitSubgraph: {
      subgraph_defs = unit_subgraph_defs;
      // 直接将单元子图赋值给主子图定义
    } break;
    case SubgraphPreparationType::kMergeUnitSubgraph: {
      // Add merged atomic subgraphs
      // Note that each merged subgraph consists of unit subgraphs with
      // continuous unit subgraph indices.
      // If we find any of the case that does not satisfy the condition,
      // we should re-implement the merging logic.
      subgraph_defs = MergeUnitSubgraphs(unit_subgraph_defs);
      // 根据连续性和其他标准合并单元子图，确保合并的子图由连续的单元子图索引组成。
    } break;
    default: {
      return absl::InternalError(absl::StrFormat(
          "Failed to create subgraph. Unsupported subgraph preparation type "
          "%s for model %s and mode %s",
          ToString(subgraph_config_.subgraph_preparation_type),
          model_spec_->path.c_str(),
          ToString(subgraph_config_.subgraph_preparation_type)));
    }
  }

  // Verify subgraphs
  {
    // unit subgraph indices in merged subgraph are continous
    // 进行验证步骤，以确保任何合并的子图中，单元子图的索引是连续的
    for (const auto& subgraph_def : subgraph_defs) {
      const int begin = *subgraph_def.unit_subgraph_indices.begin();
      const int end = *subgraph_def.unit_subgraph_indices.rbegin();
      if (end - begin != subgraph_def.unit_subgraph_indices.size() - 1) {
        return absl::InternalError(absl::StrFormat(
            "Failed to create subgraph. Unit subgraph indices in "
            "subgraph %s are not continous for model %s and mode %s",
            subgraph_def.ToString().c_str(), model_spec_->path.c_str(),
            ToString(subgraph_config_.subgraph_preparation_type)));
      }
    }
  }

  const std::string subgraph_summary =
      subgraph_config_.subgraph_preparation_type !=
              SubgraphPreparationType::kFallbackPerWorker
          ? SummarizeSubgraphs(subgraph_defs)
          : SummarizeFallbackPerWorkerSubgraphs(unit_subgraph_defs,
                                                subgraph_defs);
                                                // 生成子图的摘要

  BAND_LOG_DEBUG("Create %d subgraphs for model %s with mode %s %s",
                 subgraph_defs.size(), model_spec_->path.c_str(),
                 ToString(subgraph_config_.subgraph_preparation_type),
                 subgraph_summary.c_str());

  return std::make_pair(*model_spec_, subgraph_defs);
}

/**
 * @brief Retrieves the unit subgraphs.
 * 
 * This function populates the provided vector with the unit subgraphs.
 * A unit subgraph represents a subset of the model's operations that can be executed together on a specific worker.
 * 主要任务是根据模型规格和不同工作器的能力，划分和确认各个单元子图的操作集合。
 * @param unit_subgraphs The vector to store the unit subgraphs.
 * @return absl::Status The status of the operation. Returns absl::OkStatus() if successful, or an error status if there was a failure.
 */
absl::Status ModelAnalyzer::GetUnitSubgraphs(
    std::vector<SubgraphDef>& unit_subgraphs) {
  const int num_workers = engine_.GetNumWorkers();
  // 获取工作器数量
  unit_subgraphs.clear();
  // 清空单元子图集合

  if (!NeedFallbackSubgraph()) {
    // 如果不需要回退子图
    std::set<int> entire_ops;
    for (int i = 0; i < model_spec_->num_ops; i++) {
      entire_ops.insert(i);
      // 将所有操作添加到 entire_ops 集合中
    }

    for (WorkerId worker_id = 0; worker_id < num_workers; worker_id++) {
      if (IsWorkerValid(worker_id)) {
        unit_subgraphs.push_back({worker_id, entire_ops, {0}});
        // 遍历每个有效的工作器，并为每个工作器创建包含所有操作的子图
      }
    }
  } else {
    const int num_ops = model_spec_->num_ops;
    // 获取操作数量
    if (num_workers > 8 * sizeof(BitMask)) {
      // 检查工作器数量是否超过 BitMask 的大小限制
      return absl::InternalError(absl::StrFormat(
          "Number of workers is larger than BitMask %d", num_workers));
    }

    std::map<WorkerId, std::set<int>> op_sets_to_ignore;
    // register subgraphs for all workers
    // 为所有工作器注册子图
    for (WorkerId worker_id = 0; worker_id < num_workers; ++worker_id) {
      std::vector<SubgraphDef> worker_op_sets =
          GetSubgraphsForFallbackOps(worker_id);
          // 对每个工作器，获取特定于回退操作的子图集
      for (auto worker_and_ops : worker_op_sets) {
        if (engine_.GetWorker(worker_id)->GetDeviceFlag() == DeviceFlag::kCPU) {
          continue;
        }
        if (worker_and_ops.op_indices.size() <
            subgraph_config_.minimum_subgraph_size) {
          for (auto op : worker_and_ops.op_indices) {
            op_sets_to_ignore[worker_id].insert(op);
            // 记录因为大小不足而被忽略的操作集
          }
        }
      }
    }

    // build op_support_table
    // 构建操作支持表
    std::vector<BitMask> op_support_table(num_ops, 0U);
    std::map<WorkerId, std::set<int>> unsupported_ops;
    int unit_subgraph_index = 0;
    // TODO(BAND-62): assume that band device type targets a single processor.
    // 假设 band 设备类型只针对单个处理器
    for (WorkerId worker_id = 0; worker_id < num_workers; ++worker_id) {
      if (IsWorkerValid(worker_id)) {
        unsupported_ops[worker_id] = model_spec_->unsupported_ops.at(
            engine_.GetWorker(worker_id)->GetDeviceFlag());
            // 从 model_spec_ 中获取该工作器不支持的操作并记录在 unsupported_ops 中
      }
    }

    for (int op_index = 0; op_index < num_ops; op_index++) {
      // 遍历模型中的所有操作
      for (WorkerId worker_id = 0; worker_id < num_workers; ++worker_id) {
        // 进一步遍历所有工作器
        DeviceFlag device_flag = engine_.GetWorker(worker_id)->GetDeviceFlag();
        // 获取当前工作器的设备标志
        if (device_flag == DeviceFlag::kCPU) {
          op_support_table[op_index] |= 1 << worker_id;
          // 如果设备标志为 CPU，将当前工作器的支持设备添加到操作支持表中
          continue;
        }

        WorkerId tmp_worker_id = static_cast<int>(device_flag);
        // 如果不是 CPU，将设备标志转换为整型 tmp_worker_id

        // 处理非 CPU 设备的操作支持
        if (unsupported_ops.find(tmp_worker_id) == unsupported_ops.end() ||
            unsupported_ops.at(tmp_worker_id).find(op_index) ==
                unsupported_ops.at(tmp_worker_id).end()) {
                  // 首先检查tmp_worker_id 对应的工作器是否有记录不支持当前操作 op_index
          if (op_sets_to_ignore[tmp_worker_id].find(op_index) ==
              op_sets_to_ignore[tmp_worker_id].end()) {
                // 进一步检查是否应该忽略此操作（基于 op_sets_to_ignore 集合）
            op_support_table[op_index] |= 1 << worker_id;
            // 同样在 op_support_table 的对应位置设置支持位
          }
        }
      }
    }

    // Add unit Subgraphs.
    // 添加单元子图
    // Op indices in single unit subgraph have same support devices.
    // 单元子图中的操作索引具有相同的支持设备
    std::set<int> resolved_tensors;
    std::set<int> remaining_ops;

    for (int input_index : model_spec_->input_tensors) {
      resolved_tensors.insert(input_index);
    }

    for (int i = 0; i < num_ops; i++) {
      remaining_ops.insert(i);
    }

    while (true) {
      std::set<int> unit_subgraph_ops;
      BitMask support_workers = 0;
      // 一个位掩码，表示支持当前单元子图中所有操作的工作器

      // Find single unit subgraph ops
      // 找到单元子图操作
      while (true) {
        // Find addable ops
        // 查找可添加的操作
        // 1. resolved
        // 2. same support devices
        std::vector<int> to_add;
        for (int op_index : remaining_ops) {
          // Check the op is resolved
          if (!IsResolved(resolved_tensors, op_index)) {
            continue;
          }
          // Check the op have same support devices
          if (support_workers != 0 &&
              support_workers != op_support_table[op_index]) {
                // 通过比较 support_workers 和操作的支持工作器（从 op_support_table 中获取）
            continue;
          }
          // Set support devices using first op
          if (support_workers == 0) {
            // 如果是循环中的第一个操作，它会初始化 support_workers
            support_workers = op_support_table[op_index];
          }
          to_add.push_back(op_index);
        }
        // If there is no more ops to add, stop
        if (to_add.empty()) break;

        // Add ops which are resolved and have same support devices
        // 如果 to_add 不为空，这些操作会被正式加入到 unit_subgraph_ops 中
        unit_subgraph_ops.insert(to_add.begin(), to_add.end());

        // Delete resolved ops and add resolved tensors
        // 并且相关的操作会从 remaining_ops 中移除，表示它们已经被处理。
        for (int op_index : to_add) {
          remaining_ops.erase(remaining_ops.find(op_index));
          const std::set<int>& op_outputs =
              model_spec_->op_output_tensors[op_index];
          for (int op_output_tensor : op_outputs) {
            resolved_tensors.insert(op_output_tensor);
          }
        }
      }
      if (unit_subgraph_ops.empty()) break;
      for (WorkerId worker_id = 0; worker_id < num_workers; ++worker_id) {
        if (!IsWorkerValid(worker_id)) {
          continue;
        }
        if (support_workers.test(worker_id)) {
          unit_subgraphs.push_back(
              {worker_id, unit_subgraph_ops, {unit_subgraph_index}});
        }
      }
      unit_subgraph_index++;
    }

    if (!remaining_ops.empty()) {
      return absl::InternalError("Not empty remaining ops");
    }
    // 需要回退运算符的情况获取单元子图处理完毕
  }

// 构建子图单元后，将子图整理并映射到模型规范中
  std::set<int> unique_unit_subgraph_indices;
  // 存储所有独立单元子图索引的集合

  for (const auto& subgraph_def : unit_subgraphs) {
    unique_unit_subgraph_indices.insert(
        *subgraph_def.unit_subgraph_indices.begin());
  }

  std::vector<std::set<int>> unit_subgraph_defs;
  // 存储单元子图定义的集合
  // 其中的每个元素都是一个集合，用来存储对应索引的子图中包含的操作索引

  unit_subgraph_defs.resize(unique_unit_subgraph_indices.size());
  for (const auto& unit_subgraph_def : unit_subgraphs) {
    unit_subgraph_defs[*unit_subgraph_def.unit_subgraph_indices.begin()] =
        unit_subgraph_def.op_indices;
  }

  // TODO(widiba03304): Add error propagation logic.
  auto status = model_spec_->SetUnitSubgraphs(unit_subgraph_defs);
  // 将单元子图定义映射到模型规范中
  if (!status.ok()) {
    return status;
  }

  // 验证构建的单元子图之间的正确性和独立性
  for (const auto& lhs : unit_subgraphs) {
    for (const auto& rhs : unit_subgraphs) {
      if (&lhs == &rhs) {
        // lhs 和 rhs 分别表示当前正在比较的两个子图
        // 跳过自身比较
        continue;
      }

      if (*lhs.unit_subgraph_indices.begin() ==
          *rhs.unit_subgraph_indices.begin()) {
        if (lhs.op_indices != rhs.op_indices) {
          // 检查子图索引冲突
          return absl::InternalError(absl::StrFormat(
              "Failed to create unit subgraph. Unit subgraph with same idx %d "
              "has different operators",
              *lhs.unit_subgraph_indices.begin()));
        }
      } else {
        // 检查操作的重叠
        // 如果两个子图的索引不同，进一步检查它们是否包含任何共同的操作
        std::set<int> intersection;
        std::set_intersection(
            lhs.op_indices.begin(), lhs.op_indices.end(),
            rhs.op_indices.begin(), rhs.op_indices.end(),
            std::inserter(intersection, intersection.begin()));
            // 使用 std::set_intersection 来找出两组操作中的公共元素
        if (intersection.size()) {
          // 表明有操作重叠
          return absl::InternalError(absl::StrFormat(
              "Failed to create unit subgraph. Unit subgraph with "
              "different idx %d, %d "
              "has common operators %s",
              *lhs.unit_subgraph_indices.begin(),
              *rhs.unit_subgraph_indices.begin(),
              SetToString(intersection).c_str()));
        }
      }
    }
  }

  BAND_LOG(LogSeverity::kInternal,
           "Create %d unit subgraphs, planner requires subgraph %d",
           unique_unit_subgraph_indices.size(), NeedFallbackSubgraph());

  return absl::OkStatus();
}

/**
 * @brief A vector of SubgraphDef objects.
 * 为指定的工作器生成子图
 * This vector represents a collection of SubgraphDef objects, which define subgraphs
 * within the model. Each SubgraphDef contains information about the worker ID, the set
 * of operators in the subgraph, and any additional properties.
 */
std::vector<SubgraphDef> band::ModelAnalyzer::GetSubgraphsForFallbackOps(
    WorkerId worker_id) {
  const Worker* worker = engine_.GetWorker(worker_id);
  if (!worker) {
    BAND_LOG(LogSeverity::kWarning, "Invalied worker_id %d", worker_id);
    return {};
  }

  if (!IsWorkerValid(worker_id)) {
    return {};
  }

  if (!NeedFallbackSubgraph()) {
    std::set<int> entire_ops;
    for (int i = 0; i < model_spec_->num_ops; i++) {
      entire_ops.insert(i);
    }
    return {{worker_id, entire_ops, {0}}};
  }

  std::vector<SubgraphDef> subgraph_defs;
  const int num_ops = model_spec_->num_ops;
  const DeviceFlag device_flag = engine_.GetWorker(worker_id)->GetDeviceFlag();
  const std::set<int>& unsupported_ops =
      model_spec_->unsupported_ops.at(device_flag);
      // 根据设备标识获取不支持的操作集合

  std::set<int> cpu_worker_ids;
  for (WorkerId worker_id = 0; worker_id < engine_.GetNumWorkers();
       worker_id++) {
    if (engine_.GetWorker(worker_id)->GetDeviceFlag() == DeviceFlag::kCPU) {
      cpu_worker_ids.insert(worker_id);
      // 记录所有 CPU 工作器的 ID
    }
  }

  std::set<int> resolved_tensors;
  // 已解决的张量集合
  std::set<int> remaining_ops;
  // 未解决的操作集合
  // The basic idea is to partition this model into several disjoint
  // subgraphs. Each subgraph is not necessarily a connected graph, and no two
  // graphs have any common ops. A subgraph is either a fallback subgraph or a
  // non-fallback one, but (obviously) never both.
  // 我们的基本方案是把模型分割成若干个相互独立的子图。
  // 每个子图不必然是连通的，而且它们之间不会有共享的操作。
  // 根据定义，每个子图非是回退子图，即用于处理特定情况的备用方案，就是常规子图，两者不可兼得。
  //
  //   Subgraph1  Sbg2     Sbg3
  // |--Non-fb--|--fb--|--Non-fb-|
  //
  //       Op2 --- Op3 -- Op4
  //     /                   \
  // Op1 - Op5 --- Op6 -- Op7 - Op8
  //
  // We start from the foremost op(s) and gradually "expand" our territory of
  // ops until we have the largest subgraph possible, without going over the
  // boundary of fallback/non-fallback. After that, we remove the ops of that
  // largest subgraph and start over with the remaining ops. This process is
  // repeated until all ops have been removed.
  // 我们的方法是从模型中最开始的操作着手，逐步扩大操作的范围，直到形成一个尽可能大的子图，同时确保不将回退和非回退操作混合。
  // 完成一个最大子图后，我们将其从模型中剔除，然后对剩余的操作重复此过程，直到所有操作都被分类到子图中。

  // To make this work, we first need to keep track of the "front line" of
  // ops. This front line, together with the fallback/non-fb status of the op,
  // is used to determine whether or not we include an op in the current
  // subgraph.
  // 为了实施这一策略，我们需要先确定操作的“前线”。
  // 这里的前线，结合操作是属于回退还是非回退类别，帮助我们决定是否将某个操作纳入当前的子图。

  // The front line is denoted with the set of "resolved" tensors -- a tensor
  // is considered resolved if that tensor can be computed using external
  // inputs + previously resolved tensors. In case all input tensors of an
  // op are resolved ones, that op is regarded to be at the front line of ops
  // and thus can be put into the current subgraph (+ the fb/non-fb status
  // must match too).
  // 我们通过“已解决”张量的集合来标记前线。一个张量如果能通过外部输入加上之前已解决的张量计算得出，就被认为是已解决的。
  // 如果一个操作的所有输入张量都已解决，那么这个操作就位于操作前线，可以被加入到当前子图中（同时，它的回退/非回退状态也必须一致）。
  for (int input_index : model_spec_->input_tensors) {
    resolved_tensors.insert(input_index);
  }

  for (int i = 0; i < num_ops; i++) {
    remaining_ops.insert(i);
  }

  bool is_fallback = false;
  int unit_subgraph_idx = 0;
  while (remaining_ops.size() > 0) {
    std::set<int> operator_set;
    bool found = true;
    // Switch between device and fallback
    // 切换设备和回退
    DeviceFlag current_device = is_fallback ? DeviceFlag::kCPU : device_flag;

    // Get all op that has resolvable dependency to specific device
    // 获取所有与特定设备有可解决依赖关系的操作。
    while (found) {
      found = false;
      for (auto current_op = remaining_ops.begin();
           current_op != remaining_ops.end();) {
        int current_index = *current_op;
        bool is_op_unsupported =
            unsupported_ops.find(current_index) != unsupported_ops.end();
        if (!is_fallback == is_op_unsupported) {
          // either 1) this is a fallback op but we're making a non-fb
          // subgraph, or 2) this is a non-fb op but we're making a fb
          // subgraph, so we skip it
          // 此情况下，我们将跳过当前操作：
          // 1) 如果这是一个回退操作，但我们正尝试创建一个非回退子图；
          // 或者 2) 如果这是一个非回退操作，但我们正尝试创建一个回退子图。
          current_op++;
          continue;
        }

        // Dependency check
        if (!IsResolved(resolved_tensors, current_index)) {
          // 如果当前操作的依赖关系未解决，我们将跳过它
          current_op++;
          continue;
        }

        found = true;
        operator_set.insert(current_index);
        // 经过检查 符合条件 的操作会被添加到 operator_set 集合中

        const std::set<int>& op_outputs =
            model_spec_->op_output_tensors[current_index];
            // 不是在模型运行时动态产生的输出数据，而是模型的一个静态部分，描述了每个操作（operation）的输出张量（tensor）的索引。

        // Update dependency to include output tensors of this new op.
        // This has the effect of expanding the "front line" of ops.
        // 更新依赖，将这个新操作的输出张量纳入其中。
        // 这样做可以扩大操作的“前线”，即扩展到更多操作的影响范围。
        for (int op_output_tensor : op_outputs) {
          resolved_tensors.insert(op_output_tensor);
        }

        current_op = remaining_ops.erase(current_op);
      }
    }

    if (operator_set.size()) {
      if (current_device == DeviceFlag::kCPU &&
          device_flag != DeviceFlag::kCPU) {
            // 如果当前设备是 CPU，但设备标志不是 CPU
        for (auto cpu_worker_id : cpu_worker_ids) {
          subgraph_defs.push_back({cpu_worker_id, operator_set, {}});
          // 将操作添加到 CPU 工作器的子图中
        }
      } else {
        subgraph_defs.push_back({worker_id, operator_set, {}});
        // 将操作添加到当前工作器的子图中
      }
    }

    unit_subgraph_idx++;
    is_fallback = !is_fallback;
    // 切换 is_fallback 的状态，以便下一轮可以针对另一种类型的操作（回退或非回退）进行子图构建
  }

  return subgraph_defs;
}

/**
 * @brief A container class that stores multiple instances of the `SubgraphDef` class.
 * 主要目的是合并单元子图，形成可能更大的子图，这些子图包含在相同工作器上可连续执行的操作。
 * This class provides a dynamic array-like container that can hold multiple instances of the `SubgraphDef` class.
 * It allows for efficient insertion, deletion, and access of `SubgraphDef` objects.
 * 
 * @tparam T The type of elements stored in the vector, which in this case is `SubgraphDef`.
 */
std::vector<SubgraphDef> ModelAnalyzer::MergeUnitSubgraphs(
    const std::vector<SubgraphDef>& unit_subgraphs) {
  std::vector<SubgraphDef> result_subgraphs = unit_subgraphs;

  // Check given worker - op_indices pair is already created or not
  // 用于检查具有给定 worker_id 和操作索引集 op_indices 的子图是否已经存在于 result_subgraphs 中
  auto is_already_created = [&result_subgraphs](WorkerId worker_id,
                                                std::set<int> op_indices) {
    for (const auto& subgraph : result_subgraphs) {
      if (subgraph.worker_id == worker_id &&
          subgraph.op_indices == op_indices) {
        return true;
      }
    }
    return false;
  };

  int num_subgraphs_before_merge = unit_subgraphs.size();
  bool added = true;
  while (added) {
    added = false;
    std::vector<SubgraphDef> subgraphs_to_add;
    // 暂存这一层生成的子图
    for (const auto& prev_unit_subgraph : result_subgraphs) {
      const std::set<int> prev_outputs =
          model_spec_->GetOutputTensors(prev_unit_subgraph.op_indices);
      for (const auto& next_unit_subgraph : result_subgraphs) {
        // Prepare merged worker_id - op_indices
        const WorkerId worker_id = prev_unit_subgraph.worker_id;
        const std::set<int> next_inputs =
            model_spec_->GetPureInputTensors(next_unit_subgraph.op_indices);
        // Skip same subgraph or different device
        // 跳过相同的子图或不同的设备
        if ((&prev_unit_subgraph == &next_unit_subgraph) ||
            (prev_unit_subgraph.worker_id != next_unit_subgraph.worker_id)) {
          continue;
        }
        // Check whether prev subgraph fully resolves the next
        // 检查前一个子图是否完全解析了下一个子图
        if (!std::includes(prev_outputs.begin(), prev_outputs.end(),
                           next_inputs.begin(), next_inputs.end())) {
          continue;
        }
        // 检查 prev_unit_subgraph 的输出是否完全涵盖 next_unit_subgraph 的输入
        // 这确保了 next_unit_subgraph 所需的所有数据都可以由 prev_unit_subgraph 提供。

        std::set<int> op_indices;
        const std::set<int>& prev_op_indices = prev_unit_subgraph.op_indices;
        // 获取前一个子图的操作索引
        const std::set<int>& next_op_indices = next_unit_subgraph.op_indices;
        // 获取下一个子图的操作索引
        std::set_union(prev_op_indices.begin(), prev_op_indices.end(),
                       next_op_indices.begin(), next_op_indices.end(),
                       std::inserter(op_indices, op_indices.end()));
        // 合并两个子图的操作索引

        std::set<int> unit_subgraph_indices;
        std::set_union(
            prev_unit_subgraph.unit_subgraph_indices.begin(),
            prev_unit_subgraph.unit_subgraph_indices.end(),
            next_unit_subgraph.unit_subgraph_indices.begin(),
            next_unit_subgraph.unit_subgraph_indices.end(),
            std::inserter(unit_subgraph_indices, unit_subgraph_indices.end()));
        // 合并两个子图的单元子图索引

        // Add if not already created
        if (!is_already_created(worker_id, op_indices)) {
          subgraphs_to_add.push_back(
              {worker_id, op_indices, unit_subgraph_indices});
        }
      }
    }

    for (auto& subgraph : subgraphs_to_add) {
      if (is_already_created(subgraph.worker_id, subgraph.op_indices)) continue;
      added = true;
      result_subgraphs.push_back(subgraph);
    }
  }

  BAND_LOG(LogSeverity::kInternal, "Create %d merged subgraphs",
           result_subgraphs.size() - num_subgraphs_before_merge);

  return result_subgraphs;
}

/**
 * Checks if a fallback subgraph is needed.
 *
 * This function determines whether a fallback subgraph is needed based on the
 * value of `need_fallback_subgraph_` and the `subgraph_preparation_type` in
 * the `subgraph_config_`. If `need_fallback_subgraph_` is true and
 * `subgraph_preparation_type` is not `kNoFallbackSubgraph`, then a fallback
 * subgraph is needed.
 *
 * @return True if a fallback subgraph is needed, false otherwise.
 */
bool ModelAnalyzer::NeedFallbackSubgraph() const {
  return need_fallback_subgraph_ &&
         (subgraph_config_.subgraph_preparation_type !=
          SubgraphPreparationType::kNoFallbackSubgraph);
}

/**
 * Checks if a worker is valid.
 *
 * This function checks if the specified worker is valid by verifying if its device flag is present in the set of unavailable devices in the model specification.
 *
 * @param worker_id The ID of the worker to check.
 * @return `true` if the worker is valid, `false` otherwise.
 */
bool ModelAnalyzer::IsWorkerValid(WorkerId worker_id) const {
  return model_spec_->unavailable_devices.find(
             engine_.GetWorker(worker_id)->GetDeviceFlag()) ==
         model_spec_->unavailable_devices.end();
}

/**
 * Checks if a given operation is resolved based on the set of resolved tensors.
 *
 * @param resolved_tensors A set of resolved tensor indices.
 * @param op_index The index of the operation to check.
 * @return True if the operation is resolved, false otherwise.
 */
bool ModelAnalyzer::IsResolved(const std::set<int> resolved_tensors,
                               int op_index) const {
  const std::set<int>& op_inputs = model_spec_->op_input_tensors[op_index];
  for (int op_input_tensor : op_inputs) {
    if (resolved_tensors.find(op_input_tensor) == resolved_tensors.end()) {
      return false;
    }
  }
  return true;
}
}  // namespace band