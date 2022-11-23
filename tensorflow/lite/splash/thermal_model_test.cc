#include "tensorflow/lite/splash/thermal_model.h"

#include <gtest/gtest.h>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace impl {
namespace {

TEST(ThermalModelTest, NormalEquantion) {
  Eigen::MatrixXd x;
  Eigen::VectorXd y;

  x.conservativeResize(2, 2);
  y.conservativeResize(2, 1);

  x.row(0) << 1, 2;
  x.row(1) << 1, 3;

  y.row(0) << 1;
  y.row(1) << 2;

  auto theta = IThermalModel::GetNormalEquation<double, 2, 1>(x, y);

  EXPECT_NEAR(theta(0, 0), -1, 0.00001);
  EXPECT_NEAR(theta(1, 0), 1, 0.00001);
}

}  // namespace  
}  // namespace impl
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
