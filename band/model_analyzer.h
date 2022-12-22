#ifndef BAND_MODEL_ANALYZER_H_
#define BAND_MODEL_ANALYZER_H_

#include <map>

#include "band/context.h"

namespace Band {
class Model;

class ModelAnalyzer {
 public:
  ModelAnalyzer(const Context& context, Model* model,
                BandBackendType backend_type);

  BandStatus GetUnitSubgraphs(
      ModelId model_id, bool need_fallback_subgraph,
      // <worker id, op indices>
      std::vector<std::pair<WorkerId, std::set<size_t>>>& unit_subgraphs);

  const ModelSpec& GetModelSpec() const;

 private:
  ModelSpec model_spec_;
  const Context& context_;
  const BandBackendType target_backend_type_;
};
}  // namespace Band

#endif