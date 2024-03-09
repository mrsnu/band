#ifndef BAND_GRAPH_SHAPE_H_
#define BAND_GRAPH_SHAPE_H_

#include <vector>

#include "band/common.h"

namespace band {

struct Shape {
  DataType dtype;
  std::vector<int> dims;
};

}  // namespace band

#endif  // BAND_GRAPH_SHAPE_H_