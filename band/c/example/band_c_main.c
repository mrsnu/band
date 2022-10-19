#include "band/c/c_api.h"
#include "band/c/common.h"

#if defined(_WIN32)
#define __WINDOWS__
#endif

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#ifdef __WINDOWS__
#include <windows.h>
#else
#include <dlfcn.h>
#endif

PFN_BandAddConfig pBandAddConfig;
PFN_BandBackendGetName pBandBackendGetName;
PFN_BandBackendGetType pBandBackendGetType;
PFN_BandConfigBuilderCreate pBandConfigBuilderCreate;
PFN_BandConfigBuilderDelete pBandConfigBuilderDelete;
PFN_BandConfigCreate pBandConfigCreate;
PFN_BandConfigDelete pBandConfigDelete;
PFN_BandDeviceGetFlag pBandDeviceGetFlag;
PFN_BandDeviceGetName pBandDeviceGetName;
PFN_BandEngineCreate pBandEngineCreate;
PFN_BandEngineCreateInputTensor pBandEngineCreateInputTensor;
PFN_BandEngineCreateOutputTensor pBandEngineCreateOutputTensor;
PFN_BandEngineDelete pBandEngineDelete;
PFN_BandEngineGetNumInputTensors pBandEngineGetNumInputTensors;
PFN_BandEngineGetNumOutputTensors pBandEngineGetNumOutputTensors;
PFN_BandEngineRegisterModel pBandEngineRegisterModel;
PFN_BandEngineRequestAsync pBandEngineRequestAsync;
PFN_BandEngineRequestSync pBandEngineRequestSync;
PFN_BandEngineWait pBandEngineWait;
PFN_BandFloatArrayCreate pBandFloatArrayCreate;
PFN_BandFloatArrayFree pBandFloatArrayFree;
PFN_BandFloatArrayGetSizeInBytes pBandFloatArrayGetSizeInBytes;
PFN_BandIntArrayCopy pBandIntArrayCopy;
PFN_BandIntArrayCreate pBandIntArrayCreate;
PFN_BandIntArrayEqual pBandIntArrayEqual;
PFN_BandIntArrayEqualsArray pBandIntArrayEqualsArray;
PFN_BandIntArrayFree pBandIntArrayFree;
PFN_BandIntArrayGetSizeInBytes pBandIntArrayGetSizeInBytes;
PFN_BandModelAddFromBuffer pBandModelAddFromBuffer;
PFN_BandModelAddFromFile pBandModelAddFromFile;
PFN_BandModelCreate pBandModelCreate;
PFN_BandModelDelete pBandModelDelete;
PFN_BandTensorDelete pBandTensorDelete;
PFN_BandTensorGetBytes pBandTensorGetBytes;
PFN_BandTensorGetData pBandTensorGetData;
PFN_BandTensorGetDims pBandTensorGetDims;
PFN_BandTensorGetName pBandTensorGetName;
PFN_BandTensorGetQuantization pBandTensorGetQuantization;
PFN_BandTensorGetType pBandTensorGetType;
PFN_BandTypeGetName pBandTypeGetName;

#ifdef __WINDOWS__
#define LoadFunction(function) \
  p##function = (PFN_##function)GetProcAddress(libbandc, #function);
#else
#define LoadFunction(function) \
  p##function = (PFN_##function)dlsym(libbandc, #function);
#endif

#ifdef __WINDOWS__
void LoadBandLibraryFunctions(HMODULE libbandc) {
#else
void LoadBandLibraryFunctions(void* libbandc) {
#endif
  LoadFunction(BandAddConfig);
  LoadFunction(BandBackendGetName);
  LoadFunction(BandBackendGetType);
  LoadFunction(BandConfigBuilderCreate);
  LoadFunction(BandConfigBuilderDelete);
  LoadFunction(BandConfigCreate);
  LoadFunction(BandConfigDelete);
  LoadFunction(BandDeviceGetFlag);
  LoadFunction(BandDeviceGetName);
  LoadFunction(BandEngineCreate);
  LoadFunction(BandEngineCreateInputTensor);
  LoadFunction(BandEngineCreateOutputTensor);
  LoadFunction(BandEngineDelete);
  LoadFunction(BandEngineGetNumInputTensors);
  LoadFunction(BandEngineGetNumOutputTensors);
  LoadFunction(BandEngineRegisterModel);
  LoadFunction(BandEngineRequestAsync);
  LoadFunction(BandEngineRequestSync);
  LoadFunction(BandEngineWait);
  LoadFunction(BandFloatArrayCreate);
  LoadFunction(BandFloatArrayFree);
  LoadFunction(BandFloatArrayGetSizeInBytes);
  LoadFunction(BandIntArrayCopy);
  LoadFunction(BandIntArrayCreate);
  LoadFunction(BandIntArrayEqual);
  LoadFunction(BandIntArrayEqualsArray);
  LoadFunction(BandIntArrayFree);
  LoadFunction(BandIntArrayGetSizeInBytes);
  LoadFunction(BandModelAddFromBuffer);
  LoadFunction(BandModelAddFromFile);
  LoadFunction(BandModelCreate);
  LoadFunction(BandModelDelete);
  LoadFunction(BandTensorDelete);
  LoadFunction(BandTensorGetBytes);
  LoadFunction(BandTensorGetData);
  LoadFunction(BandTensorGetDims);
  LoadFunction(BandTensorGetName);
  LoadFunction(BandTensorGetQuantization);
  LoadFunction(BandTensorGetType);
  LoadFunction(BandTypeGetName);
}

bool LoadBandLibrary() {
#ifdef __WINDOWS__
  HMODULE libbandc = LoadLibraryA("band_c.dll");
  if (libbandc) {
    LoadBandLibraryFunctions(libbandc);
    return true;
  } else {
    DWORD error_code = GetLastError();
    fprintf(stderr,
            "Cannnot open Band C Library oon this device, error code - %d\n",
            error_code);
    return false;
  }
#else
  void* libbandc = NULL;
  libbandc = dlopen("libband_c.so", RTLD_NOW | RTLD_LOCAL);
  if (libbandc) {
    LoadBandLibraryFunctions(libbandc);
    return true;
  }
#endif
}

int main() {
  bool success = LoadBandLibrary();
  if (success) {
    printf("Loaded!\n");
  } else {
    printf("Load failed.\n");
    return -1;
  }
  BandConfigBuilder* b = pBandConfigBuilderCreate();
  printf("BandConfigBuilder\n");
  pBandAddConfig(b, BAND_PLANNER_LOG_PATH, /*count=*/1, "log.tsv");
  printf("BandAddConfig, BAND_PLANNER_LOG_PATH\n");
  pBandAddConfig(b, BAND_PLANNER_SCHEDULERS, /*count=*/1, kBandRoundRobin);
  printf("BandAddConfig, BAND_PLANNER_SCHEDULERS\n");
  pBandAddConfig(b, BAND_MINIMUM_SUBGRAPH_SIZE, /*count=*/1, 7);
  printf("BandAddConfig, BAND_MINIMUM_SUBGRAPH_SIZE\n");
  pBandAddConfig(b, BAND_SUBGRAPH_PREPARATION_TYPE, /*count=*/1,
                 kBandMergeUnitSubgraph);
  printf("BandAddConfig, BAND_SUBGRAPH_PREPARATION_TYPE\n");
  pBandAddConfig(b, BAND_CPU_MASK, /*count=*/1, kBandAll);
  printf("BandAddConfig, BAND_CPU_MASK\n");
  pBandAddConfig(b, BAND_PLANNER_CPU_MASK, /*count=*/1, kBandPrimary);
  printf("BandAddConfig, BAND_PLANNER_CPU_MASK\n");
  pBandAddConfig(b, BAND_WORKER_WORKERS, /*count=*/2, kBandCPU, kBandCPU);
  printf("BandAddConfig, BAND_WORKER_WORKERS\n");
  pBandAddConfig(b, BAND_WORKER_NUM_THREADS, /*count=*/2, 3, 4);
  printf("BandAddConfig, BAND_WORKER_NUM_THREADS\n");
  pBandAddConfig(b, BAND_WORKER_CPU_MASKS, /*count=*/2, kBandBig, kBandLittle);
  printf("BandAddConfig, BAND_WORKER_CPU_MASKS\n");
  pBandAddConfig(b, BAND_PROFILE_SMOOTHING_FACTOR, /*count=*/1, 0.1f);
  printf("BandAddConfig, BAND_PROFILE_SMOOTHING_FACTOR\n");
  pBandAddConfig(b, BAND_PROFILE_DATA_PATH, /*count=*/1,
                 "band/testdata/profile.json");
  printf("BandAddConfig, BAND_PROFILE_DATA_PATH\n");
  pBandAddConfig(b, BAND_PROFILE_ONLINE, /*count=*/1, true);
  printf("BandAddConfig, BAND_PROFILE_ONLINE\n");
  pBandAddConfig(b, BAND_PROFILE_NUM_WARMUPS, /*count=*/1, 1);
  printf("BandAddConfig, BAND_PROFILE_NUM_WARMUPS\n");
  pBandAddConfig(b, BAND_PROFILE_NUM_RUNS, /*count=*/1, 1);
  printf("BandAddConfig, BAND_PROFILE_NUM_RUNS\n");
  pBandAddConfig(b, BAND_WORKER_ALLOW_WORKSTEAL, /*count=*/1, true);
  printf("BandAddConfig, BAND_WORKER_ALLOW_WORKSTEAL\n");
  pBandAddConfig(b, BAND_WORKER_AVAILABILITY_CHECK_INTERVAL_MS, /*count=*/1,
                 30000);
  printf("BandAddConfig, BAND_WORKER_AVAILABILITY_CHECK_INTERVAL_MS\n");
  pBandAddConfig(b, BAND_PLANNER_SCHEDULE_WINDOW_SIZE, /*count=*/1, 10);
  printf("BandAddConfig, BAND_PLANNER_SCHEDULE_WINDOW_SIZE\n");
  BandConfig* config = pBandConfigCreate(b);
  printf("BandConfigCreate\n");

  BandModel* model = pBandModelCreate();
  printf("BandModelCreate\n");
  pBandModelAddFromFile(model, kBandTfLite, "band/testdata/add.bin");
  printf("BandModelAddFromFile\n");

  BandEngine* engine = pBandEngineCreate(config);
  printf("BandEngineCreate\n");
  pBandEngineRegisterModel(engine, model);
  printf("BandEngineRegisterModel\n");
  pBandEngineGetNumInputTensors(engine, model);
  printf("BandGetNumInputTensors\n");
  pBandEngineGetNumOutputTensors(engine, model);
  printf("BandGetNumOutputTensors\n");

  BandTensor* input_tensor = pBandEngineCreateInputTensor(engine, model, 0);
  printf("BandEngineCreateInputTensor\n");
  BandTensor* output_tensor = pBandEngineCreateOutputTensor(engine, model, 0);
  printf("BandEngineCreateOutputTensor\n");

  float input[] = {1.f, 3.f};
  memcpy(pBandTensorGetData(input_tensor), input, 2 * sizeof(float));
  printf("BandTensorGetData\n");
  pBandEngineRequestSync(engine, model, &input_tensor, &output_tensor);
  printf("BandEngineRequestSync\n");

  if (((float*)pBandTensorGetData(output_tensor))[0] == 3.f &&
      ((float*)pBandTensorGetData(output_tensor))[1] == 9.f) {
    printf("Success!\n");
  }

  pBandEngineDelete(engine);
  pBandTensorDelete(input_tensor);
  pBandTensorDelete(output_tensor);
  pBandConfigDelete(config);
  pBandModelDelete(model);
  return 0;
}