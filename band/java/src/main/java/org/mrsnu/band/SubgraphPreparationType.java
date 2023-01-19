package org.mrsnu.band;

public enum SubgraphPreparationType {
  NO_FALLBACK_SUBGRAPH(0),
  FALLBACK_PER_WORKER(1),
  UNIT_SUBGRAPH(2),
  MERGE_UNIT_SUBGRAPH(3);
  
  private final int value;
  SubgraphPreparationType(int value) {
    this.value = value;
  }
  
  int getValue() {
    return value;
  }
}
