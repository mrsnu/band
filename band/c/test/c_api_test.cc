#include "band/c/c_api.h"

#include <gtest/gtest.h>
#include <stdarg.h>
#include <stdint.h>

#include <array>
#include <fstream>
#include <vector>

namespace Band {
TEST(CApi, ConfigLoad) {
  BandConfigBuilder* b = BandConfigBuilderCreate();
  BandAddConfig(b, BAND_PLANNER_LOG_PATH, /*count=*/1, "band/testdata/log.csv");
  BandAddConfig(b, BAND_PLANNER_SCHEDULERS, /*count=*/1, kBandRoundRobin);
  BandAddConfig(b, BAND_MINIMUM_SUBGRAPH_SIZE, /*count=*/1, 7);
  BandAddConfig(b, BAND_SUBGRAPH_PREPARATION_TYPE, /*count=*/1, kBandMergeUnitSubgraph);
  BandAddConfig(b, BAND_CPU_MASK, /*count=*/1, kBandAll);
  BandAddConfig(b, BAND_PLANNER_CPU_MASK, /*count=*/1, kBandPrimary);
  BandAddConfig(b, BAND_WORKER_WORKERS, /*count=*/2, kBandCPU, kBandCPU);
  BandAddConfig(b, BAND_WORKER_NUM_THREADS, /*count=*/2, 3, 4);
  BandAddConfig(b, BAND_WORKER_CPU_MASKS, /*count=*/2, kBandBig, kBandLittle);
  BandAddConfig(b, BAND_PROFILE_SMOOTHING_FACTOR, /*count=*/1, 0.1f);
  BandAddConfig(b, BAND_PROFILE_DATA_PATH, /*count=*/1,"band/testdata/profile.json");
  BandAddConfig(b, BAND_PROFILE_ONLINE, /*count=*/1, true);
  BandAddConfig(b, BAND_PROFILE_NUM_WARMUPS, /*count=*/1, 1);
  BandAddConfig(b, BAND_PROFILE_NUM_RUNS, /*count=*/1, 1);
  BandAddConfig(b, BAND_WORKER_ALLOW_WORKSTEAL, /*count=*/1, true);
  BandAddConfig(b, BAND_WORKER_AVAILABILITY_CHECK_INTERVAL_MS, /*count=*/1, 30000);
  BandAddConfig(b, BAND_PLANNER_SCHEDULE_WINDOW_SIZE, /*count=*/1, 10);
  BandConfig* config = BandConfigCreate(b);
  EXPECT_NE(config, nullptr);
  BandConfigDelete(config);
}

TEST(CApi, ModelLoad) {
  BandModel* model = BandModelCreate();
  EXPECT_NE(model, nullptr);
  EXPECT_EQ(BandModelAddFromFile(model, kBandTfLite, "band/testdata/add.bin"),
            kBandOk);
  BandModelDelete(model);
}

TEST(CApi, EngineSimpleInvoke) {
  BandConfigBuilder* b = BandConfigBuilderCreate();
  BandAddConfig(b, BAND_PLANNER_LOG_PATH, /*count=*/1, "band/testdata/log.csv");
  BandAddConfig(b, BAND_PLANNER_SCHEDULERS, /*count=*/1, kBandRoundRobin);
  BandAddConfig(b, BAND_MINIMUM_SUBGRAPH_SIZE, /*count=*/1, 7);
  BandAddConfig(b, BAND_SUBGRAPH_PREPARATION_TYPE, /*count=*/1, kBandMergeUnitSubgraph);
  BandAddConfig(b, BAND_CPU_MASK, /*count=*/1, kBandAll);
  BandAddConfig(b, BAND_PLANNER_CPU_MASK, /*count=*/1, kBandPrimary);
  BandAddConfig(b, BAND_WORKER_WORKERS, /*count=*/2, kBandCPU, kBandCPU);
  BandAddConfig(b, BAND_WORKER_NUM_THREADS, /*count=*/2, 3, 4);
  BandAddConfig(b, BAND_WORKER_CPU_MASKS, /*count=*/2, kBandBig, kBandLittle);
  BandAddConfig(b, BAND_PROFILE_SMOOTHING_FACTOR, /*count=*/1, 0.1f);
  BandAddConfig(b, BAND_PROFILE_DATA_PATH, /*count=*/1, "band/testdata/profile.json");
  BandAddConfig(b, BAND_PROFILE_ONLINE, /*count=*/1, true);
  BandAddConfig(b, BAND_PROFILE_NUM_WARMUPS, /*count=*/1, 1);
  BandAddConfig(b, BAND_PROFILE_NUM_RUNS, /*count=*/1, 1);
  BandAddConfig(b, BAND_WORKER_ALLOW_WORKSTEAL, /*count=*/1, true);
  BandAddConfig(b, BAND_WORKER_AVAILABILITY_CHECK_INTERVAL_MS, /*count=*/1, 30000);
  BandAddConfig(b, BAND_PLANNER_SCHEDULE_WINDOW_SIZE, /*count=*/1, 10);
  BandConfig* config = BandConfigCreate(b);
  EXPECT_NE(config, nullptr);

  BandModel* model = BandModelCreate();
  EXPECT_NE(model, nullptr);
  EXPECT_EQ(BandModelAddFromFile(model, kBandTfLite, "band/testdata/add.bin"),
            kBandOk);

  BandEngine* engine = BandEngineCreate(config);
  EXPECT_NE(engine, nullptr);
  EXPECT_EQ(BandEngineRegisterModel(engine, model), kBandOk);
  EXPECT_EQ(BandEngineGetNumInputTensors(engine, model), 1);
  EXPECT_EQ(BandEngineGetNumOutputTensors(engine, model), 1);

  BandTensor* input_tensor = BandEngineCreateInputTensor(engine, model, 0);
  BandTensor* output_tensor = BandEngineCreateOutputTensor(engine, model, 0);

  std::array<float, 2> input = {1.f, 3.f};
  memcpy(BandTensorGetData(input_tensor), input.data(),
         input.size() * sizeof(float));
  EXPECT_EQ(BandEngineRequestSync(engine, model, &input_tensor, &output_tensor),
            kBandOk);

  EXPECT_EQ(reinterpret_cast<float*>(BandTensorGetData(output_tensor))[0], 3.f);
  EXPECT_EQ(reinterpret_cast<float*>(BandTensorGetData(output_tensor))[1], 9.f);

  BandEngineDelete(engine);
  BandTensorDelete(input_tensor);
  BandTensorDelete(output_tensor);
  BandConfigDelete(config);
  BandModelDelete(model);
}

}   // namespace Band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
