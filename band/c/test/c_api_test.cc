#include <gtest/gtest.h>
#include <stdarg.h>
#include <stdint.h>

#include <array>
#include <fstream>
#include <vector>

#include "band/c/c_api.h"

namespace Band {
TEST(CApi, ConfigLoad) {
  BandConfig *config = BandConfigCreateFromFile("band/testdata/config.json");
  EXPECT_NE(config, nullptr);
  BandConfigDelete(config);
}

TEST(CApi, ModelLoad) {
  BandModel *model = BandModelCreate();
  EXPECT_NE(model, nullptr);
  EXPECT_EQ(BandModelAddFromFile(model, kBandTfLite, "band/testdata/add.bin"),
            kBandOk);
  BandModelDelete(model);
}

TEST(CApi, EngineSimpleInvoke) {
  BandConfig *config = BandConfigCreateFromFile("band/testdata/config.json");
  EXPECT_NE(config, nullptr);

  BandModel *model = BandModelCreate();
  EXPECT_NE(model, nullptr);
  EXPECT_EQ(BandModelAddFromFile(model, kBandTfLite, "band/testdata/add.bin"),
            kBandOk);

  BandEngine *engine = BandEngineCreate(config);
  EXPECT_NE(engine, nullptr);
  EXPECT_EQ(BandEngineRegisterModel(engine, model), kBandOk);
  EXPECT_EQ(BandEngineGetNumInputTensors(engine, model), 1);
  EXPECT_EQ(BandEngineGetNumOutputTensors(engine, model), 1);

  BandTensor *input_tensor = BandEngineCreateInputTensor(engine, model, 0);
  BandTensor *output_tensor = BandEngineCreateOutputTensor(engine, model, 0);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(BandTensorGetData(input_tensor), input.data(),
         input.size() * sizeof(float));
  EXPECT_EQ(BandEngineRequestSync(engine, model, &input_tensor, &output_tensor),
            kBandOk);

  EXPECT_EQ(reinterpret_cast<float *>(BandTensorGetData(output_tensor))[0],
            3.f);
  EXPECT_EQ(reinterpret_cast<float *>(BandTensorGetData(output_tensor))[1],
            9.f);

  BandEngineDelete(engine);
  BandTensorDelete(input_tensor);
  BandTensorDelete(output_tensor);
  BandConfigDelete(config);
  BandModelDelete(model);
}

} // namespace Band

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
