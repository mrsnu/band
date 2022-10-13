#include "band/common.h"
namespace Band {
SubgraphKey::SubgraphKey() {}
// special case - entire model subgraph
SubgraphKey::SubgraphKey(ModelId model_id, WorkerId worker_id)
    : model_id(model_id), worker_id(worker_id) {}
SubgraphKey::SubgraphKey(ModelId model_id, WorkerId worker_id,
                         std::set<int> input_ops, std::set<int> output_ops)
    : model_id(model_id),
      worker_id(worker_id),
      input_ops(input_ops),
      output_ops(output_ops) {}

bool SubgraphKey::operator<(const SubgraphKey& key) const {
  if (model_id != key.GetModelId()) {
    return model_id < key.GetModelId();
  }

  if (worker_id != key.GetWorkerId()) {
    return worker_id < key.GetWorkerId();
  }

  if (input_ops != key.input_ops) {
    return input_ops < key.input_ops;
  }

  return output_ops < key.output_ops;
}

bool SubgraphKey::operator==(const SubgraphKey& key) const {
  return (model_id == key.GetModelId()) && (worker_id == key.GetWorkerId()) &&
         (input_ops == key.input_ops) && (output_ops == key.output_ops);
}

std::string IndexSetToString(const std::set<int>& indices) {
  std::string result;
  for (const int& index : indices) {
    result += std::to_string(index) + ",";
  }
  result.pop_back();
  return result;
}

std::string SubgraphKey::GetInputOpsString() const {
  return IndexSetToString(input_ops);
}

std::string SubgraphKey::GetOutputOpsString() const {
  return IndexSetToString(output_ops);
}

bool SubgraphKey::IsValid() const {
  return (model_id != -1) && (worker_id != -1);
}

std::size_t SubgraphHash::operator()(const SubgraphKey& p) const {
  auto hash_func = std::hash<int>();
  std::size_t hash = hash_func(p.GetModelId()) ^ hash_func(p.GetWorkerId());

  auto hash_set = [hash_func, &hash](const std::set<int>& set) {
    for (int e : set) hash ^= hash_func(e);
  };

  hash_set(p.GetInputOps());
  hash_set(p.GetOutputOps());

  return hash;
}

std::size_t PairHash::operator()(const std::pair<int, std::set<int>>& p) const {
  auto hash_func = std::hash<int>();
  std::size_t hash = hash_func(p.first);
  for (int e : p.second) {
    hash ^= hash_func(e);
  }
  return hash;
}

}  // namespace Band