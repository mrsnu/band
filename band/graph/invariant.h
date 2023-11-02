#ifndef BAND_MODEL_GRAPH_INVARIANT_H_
#define BAND_MODEL_GRAPH_INVARIANT_H_

#include "absl/status/status.h"

namespace band {

class GraphBuilder;

class Invariant {
 public:
  virtual bool Check(const GraphBuilder& graph) const = 0;
};

class NoCycleInvariant : public Invariant {
 public:
  bool Check(const GraphBuilder& graph) const override;
};

class NoIsolatedNodeInvariant : public Invariant {
 public:
  bool Check(const GraphBuilder& graph) const override;
};

class NoDuplicateEdgeInvariant : public Invariant {
 public:
  bool Check(const GraphBuilder& graph) const override;
};

class NoMismatchedEdgeInvariant : public Invariant {
 public:
  bool Check(const GraphBuilder& graph) const override;
};

}  // namespace band

#endif  // BAND_MODEL_GRAPH_INVARIANT_H_