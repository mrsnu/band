#include "band/server/backend/tensorrt/trt_loader.h"

#include <gtest/gtest.h>

namespace band {
namespace server {
namespace trt {
namespace test {

TEST(TensorRTLoaderTest, LoadTest) {
  TensorRTLoader loader("libnvinfer.so");
  EXPECT_TRUE(loader.IsInitialized());
}

}  // namespace test
}  // namespace trt
}  // namespace server
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}