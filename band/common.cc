#include "band/common.h"

#include "common.h"
namespace Band {
SubgraphKey::SubgraphKey() {}
// special case - entire model subgraph
SubgraphKey::SubgraphKey(ModelId model_id, WorkerId worker_id,
                         std::set<int> unit_indices_set)
    : model_id(model_id), worker_id(worker_id) {
  for (int unit_index : unit_indices_set) {
    unit_indices.set(unit_index);
  }
}

bool SubgraphKey::operator<(const SubgraphKey& key) const {
  if (model_id != key.GetModelId()) {
    return model_id < key.GetModelId();
  }

  if (worker_id != key.GetWorkerId()) {
    return worker_id < key.GetWorkerId();
  }

  return unit_indices.to_ullong() < key.unit_indices.to_ullong();
}

bool SubgraphKey::operator==(const SubgraphKey& key) const {
  return (model_id == key.GetModelId()) && (worker_id == key.GetWorkerId()) &&
         (unit_indices == key.unit_indices);
}

bool SubgraphKey::operator!=(const SubgraphKey& key) const {
  return !(*this == key);
}

std::string IndexSetToString(const std::set<int>& indices) {
  std::string result;
  if (indices.size()) {
    for (const int& index : indices) {
      result += std::to_string(index) + ",";
    }
    result.pop_back();
  }
  return result;
}

const BitMask& SubgraphKey::GetUnitIndices() const { return unit_indices; }

std::set<int> SubgraphKey::GetUnitIndicesSet() const {
  std::set<int> indices;
  for (size_t i = 0; i < unit_indices.size(); i++) {
    if (unit_indices.test(i)) {
      indices.insert(i);
    }
  }
  return indices;
}

std::string SubgraphKey::GetUnitIndicesString() const {
  return IndexSetToString(GetUnitIndicesSet());
}

std::string SubgraphKey::ToString() const {
  return "Model id " + std::to_string(model_id) + " Worker id " +
         std::to_string(worker_id) + " Unit indices (" +
         GetUnitIndicesString() + ")";
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

  hash_set(p.GetUnitIndicesSet());

  return hash;
}

std::size_t CacheHash::operator()(const std::pair<int, BitMask>& p) const {
  auto hash_func = std::hash<int>();
  return hash_func(p.first) ^ hash_func(p.second.to_ullong());
}
}  // namespace Band