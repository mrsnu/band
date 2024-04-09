#ifndef BAND_MODEL_ANALYZER_H_
#define BAND_MODEL_ANALYZER_H_

#include <map>
#include <memory>

#include "band/engine_interface.h"
#include "band/model_spec.h"

#include "absl/status/statusor.h"

namespace band {
class Model;

struct SubgraphDef {
  WorkerId worker_id;
  std::set<int> op_indices;
  // teh op indices of the fallback ops in the subgraph.
  std::set<int> unit_subgraph_indices;
  std::string ToString() const;
};

std::string SetToString(const std::set<int>& set);
std::string SummarizeSubgraphs(const std::vector<SubgraphDef>& subgraph_defs);
std::string SummarizeFallbackPerWorkerSubgraphs(
    const std::vector<SubgraphDef>& unit_subgraph_defs,
    const std::vector<SubgraphDef>& subgraph_defs);

class ModelAnalyzer {
 public:
  ModelAnalyzer(const IEngine& engine, bool need_subgraph,
                SubgraphConfig subgraph_config, Model* model,
                BackendType backend_type);

  absl::StatusOr<std::pair<ModelSpec, std::vector<SubgraphDef>>> CreateSubgraphs();

 private:
  // A model is partitioned into unit subgraphs.
  // We assign an index to each unit subgraph, and the unit subgraph indices are
  // topologically sorted. Note that there can be better way to assign unit
  // subgraph indices if there exists any unit subgraphs that can be executed in
  // parallel.
  // 我们将一个模型分解成了若干个单元子图，并为每个子图指派了一个唯一的编号。
  // 这些编号是经过拓扑排序的，这意味着它们的排列顺序考虑了子图间的依赖关系。
  // 值得一提的是，如果某些单元子图可以同时执行，我们还有更优的编号方案以支持这种并行处理。
  absl::Status GetUnitSubgraphs(std::vector<SubgraphDef>& unit_subgraphs);
  // Generate subgraphs for fallback ops in provided model
  // This does not provides unit indices with a SubgraphDef
  // 在给定模型中为备选操作生成子图，但这个过程并不会为这些子图定义中的每个单元分配索引。
  std::vector<SubgraphDef> GetSubgraphsForFallbackOps(WorkerId worker_id);
  std::vector<SubgraphDef> MergeUnitSubgraphs(
      const std::vector<SubgraphDef>& unit_subgraphs);

  bool NeedFallbackSubgraph() const;
  bool IsWorkerValid(WorkerId worker_id) const;
  bool IsResolved(const std::set<int> resolved_tensors, int op_index) const;

  const IEngine& engine_;
  const bool need_fallback_subgraph_;
  const SubgraphConfig subgraph_config_;
  const BackendType backend_type_;
  std::shared_ptr<ModelSpec> model_spec_;
};
}  // namespace band

#endif