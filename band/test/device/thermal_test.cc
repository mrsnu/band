#include <gtest/gtest.h>

#include "band/device/thermal.h"

namespace band {
namespace test {

TEST(BandTestThermalTest, ThermalTest) {
  
}

}
}

int main(int argc, char** argv) {
#ifdef __ANDROID__
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#endif  // __ANDROID__
  return 0;
}