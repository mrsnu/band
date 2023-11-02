#ifndef BAND_MODEL_ANALYZER_H_
#define BAND_MODEL_ANALYZER_H_

#include <map>
#include <memory>

#include "absl/status/statusor.h"
#include "band/engine_interface.h"
#include "band/model_spec.h"

namespace band {
class Model;

struct SubgraphDef {
  WorkerId worker_id;
  std::set<int> op_indices;
  std::set<int> unit_subgraph_indices;
  std::string ToString() const;
};

std::string SetToString(const std::set<int>& set);
std::string SummarizeSubgraphs(const std::vector<SubgraphDef>& subgraph_defs);

class ModelAnalyzer {
 public:
  ModelAnalyzer(const IEngine& engine, bool need_subgraph,
                SubgraphConfig subgraph_config, Model* model,
                BackendType backend_type);

  absl::StatusOr<std::pair<ModelSpec, std::vector<SubgraphDef>>>
  CreateSubgraphs();

 private:
  // A model is partitioned into unit subgraphs.
  // We assign an index to each unit subgraph, and the unit subgraph indices are
  // topologically sorted. Note that there can be better way to assign unit
  // subgraph indices if there exists any unit subgraphs that can be executed in
  // parallel.
  absl::Status GetUnitSubgraphs(std::vector<SubgraphDef>& unit_subgraphs);
  // Generate subgraphs for fallback ops in provided model
  // This does not provides unit indices with a SubgraphDef
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