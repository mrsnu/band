package org.mrsnu.band;

public enum SubgraphPreparationType {
  NO_FALLBACK_SUBGRAPH(0),
  UNIT_SUBGRAPH(1),
  MERGE_UNIT_SUBGRAPH(2);
  
  private final int value;
  SubgraphPreparationType(int value) {
    this.value = value;
  }
  
  int getValue() {
    return value;
  }
}
