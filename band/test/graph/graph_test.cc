#include "band/time.h"

#include "band/graph/graph.h"
#include "band/graph/graph_builder.h"
#include "band/model.h"

#include <gtest/gtest.h>

namespace band {
namespace test {

TEST(GraphTest, ModelNodeTest) {
  Model model_1;
  auto status =
      model_1.FromPath(BackendType::kTfLite, "band/test/data/add.tflite");
  EXPECT_EQ(status, absl::OkStatus());

  Model model_2;
  status = model_2.FromPath(BackendType::kTfLite, "band/test/data/add.tflite");
  EXPECT_EQ(status, absl::OkStatus());

  Model model_3;
  status = model_3.FromPath(BackendType::kTfLite, "band/test/data/add.tflite");
  EXPECT_EQ(status, absl::OkStatus());

  Model model_4;
  status = model_4.FromPath(BackendType::kTfLite, "band/test/data/add.tflite");
  EXPECT_EQ(status, absl::OkStatus());

  GraphBuilder b("test");
  auto input_node = b.GetEntryNode();
  auto node_1 = ModelOp(model_1, input_node, "model_1");
  auto node_2 = ModelOp(model_2, node_1, "model_2");
  auto node_3 = ModelOp(model_3, node_1, "model_3");
  auto node_4 = ModelOp(model_4, node_1, "model_4");
  auto graph_or_status = b.Build();
  EXPECT_EQ(graph_or_status.status(), absl::OkStatus());
  auto graph = graph_or_status.value();
  std::cout << graph.GetGraphVizText();
}

TEST(GraphTest, ModelNodeFromPathTest) {
  GraphBuilder b("test");
  auto input_node = b.GetEntryNode();
  auto node_1 = ModelOp(BackendType::kTfLite, "band/test/data/add.tflite",
                        input_node, "model_1");
  auto node_2 = ModelOp(BackendType::kTfLite, "band/test/data/add.tflite",
                        node_1, "model_2");
  auto node_3 = ModelOp(BackendType::kTfLite, "band/test/data/add.tflite",
                        node_1, "model_3");
  auto node_4 = ModelOp(BackendType::kTfLite, "band/test/data/add.tflite",
                        node_1, "model_4");
  auto node_5 =
      BasicOp([](Tensors inputs) { return inputs; }, node_3, "basic_1");
  auto graph_or_status = b.Build();
  EXPECT_EQ(graph_or_status.status(), absl::OkStatus());
  auto graph = graph_or_status.value();
  std::cout << graph.GetGraphVizText();
}

}  // namespace test
}  // namespace band
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
