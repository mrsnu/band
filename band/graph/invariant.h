#ifndef BAND_MODEL_GRAPH_INVARIANT_H_
#define BAND_MODEL_GRAPH_INVARIANT_H_

#include "absl/status/status.h"

namespace band {

class GraphBuilder;

class Invariant {
 public:
  virtual bool Check(const GraphBuilder graph) const = 0;
};

class NoCycle : public Invariant {
 public:
  bool Check(const GraphBuilder graph) const override;
};

class NoIsolatedNode : public Invariant {
 public:
  bool Check(const GraphBuilder graph) const override;
};

class NoDuplicateEdge : public Invariant {
 public:
  bool Check(const GraphBuilder graph) const override;
};

class NoMismatchedEdge : public Invariant {
 public:
  bool Check(const GraphBuilder graph) const override;
};

}  // namespace band

#endif  // BAND_MODEL_GRAPH_INVARIANT_H_