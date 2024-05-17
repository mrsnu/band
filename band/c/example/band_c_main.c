/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "band/c/c_api.h"

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
PFN_BandConfigBuilderCreate pBandConfigBuilderCreate;
PFN_BandConfigBuilderDelete pBandConfigBuilderDelete;
PFN_BandConfigCreate pBandConfigCreate;
PFN_BandConfigDelete pBandConfigDelete;
PFN_BandEngineCreate pBandEngineCreate;
PFN_BandEngineCreateInputTensor pBandEngineCreateInputTensor;
PFN_BandEngineCreateOutputTensor pBandEngineCreateOutputTensor;
PFN_BandEngineDelete pBandEngineDelete;
PFN_BandEngineGetNumInputTensors pBandEngineGetNumInputTensors;
PFN_BandEngineGetNumOutputTensors pBandEngineGetNumOutputTensors;
PFN_BandEngineGetNumWorkers pBandEngineGetNumWorkers;
PFN_BandEngineGetWorkerDevice pBandEngineGetWorkerDevice;
PFN_BandEngineRegisterModel pBandEngineRegisterModel;
PFN_BandEngineRequestAsync pBandEngineRequestAsync;
PFN_BandEngineRequestSync pBandEngineRequestSync;
PFN_BandEngineRequestAsyncOptions pBandEngineRequestAsyncOptions;
PFN_BandEngineRequestSyncOptions pBandEngineRequestSyncOptions;
PFN_BandEngineWait pBandEngineWait;
PFN_BandEngineSetOnEndRequest pBandEngineSetOnEndRequest;
PFN_BandModelAddFromBuffer pBandModelAddFromBuffer;
PFN_BandModelAddFromFile pBandModelAddFromFile;
PFN_BandModelCreate pBandModelCreate;
PFN_BandModelDelete pBandModelDelete;
PFN_BandTensorDelete pBandTensorDelete;
PFN_BandTensorGetBytes pBandTensorGetBytes;
PFN_BandTensorGetData pBandTensorGetData;
PFN_BandTensorGetDims pBandTensorGetDims;
PFN_BandTensorGetNumDims pBandTensorGetNumDims;
PFN_BandTensorGetName pBandTensorGetName;
PFN_BandTensorGetQuantizationType pBandTensorGetQuantizationType;
PFN_BandTensorGetQuantizationParams pBandTensorGetQuantizationParams;
PFN_BandTensorGetType pBandTensorGetType;

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
  LoadFunction(BandConfigBuilderCreate);
  LoadFunction(BandConfigBuilderDelete);
  LoadFunction(BandConfigCreate);
  LoadFunction(BandConfigDelete);
  LoadFunction(BandEngineCreate);
  LoadFunction(BandEngineCreateInputTensor);
  LoadFunction(BandEngineCreateOutputTensor);
  LoadFunction(BandEngineDelete);
  LoadFunction(BandEngineGetNumInputTensors);
  LoadFunction(BandEngineGetNumOutputTensors);
  LoadFunction(BandEngineGetNumWorkers);
  LoadFunction(BandEngineGetWorkerDevice);
  LoadFunction(BandEngineRegisterModel);
  LoadFunction(BandEngineRequestAsync);
  LoadFunction(BandEngineRequestSync);
  LoadFunction(BandEngineRequestAsyncOptions);
  LoadFunction(BandEngineRequestSyncOptions);
  LoadFunction(BandEngineWait);
  LoadFunction(BandEngineSetOnEndRequest);
  LoadFunction(BandModelAddFromBuffer);
  LoadFunction(BandModelAddFromFile);
  LoadFunction(BandModelCreate);
  LoadFunction(BandModelDelete);
  LoadFunction(BandTensorDelete);
  LoadFunction(BandTensorGetBytes);
  LoadFunction(BandTensorGetData);
  LoadFunction(BandTensorGetDims);
  LoadFunction(BandTensorGetNumDims);
  LoadFunction(BandTensorGetName);
  LoadFunction(BandTensorGetQuantizationType);
  LoadFunction(BandTensorGetQuantizationParams);
  LoadFunction(BandTensorGetType);
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
            "Cannnot open Band C Library on this device, error code - %d\n",
            error_code);
    return false;
  }
#else
  void* libbandc = NULL;
  libbandc = dlopen("libband_c.so", RTLD_NOW | RTLD_LOCAL);
  if (libbandc) {
    LoadBandLibraryFunctions(libbandc);
  }
#endif
  return true;
}

void on_end_request(void* user_data, int job_id, BandStatus status) {
  if (job_id == 0 && status == kBandOk) {
    (*(int*)(user_data))++;
  }
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
  pBandAddConfig(b, BAND_PLANNER_LOG_PATH, /*count=*/1, "log.json");
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
                 "band/test/data/profile.json");
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
  pBandModelAddFromFile(model, kBandTfLite, "band/test/data/add.tflite");
  printf("BandModelAddFromFile\n");

  BandEngine* engine = pBandEngineCreate(config);
  printf("BandEngineCreate\n");
  pBandEngineRegisterModel(engine, model);
  printf("BandEngineRegisterModel\n");
  pBandEngineGetNumInputTensors(engine, model);
  printf("BandGetNumInputTensors\n");
  int num_outputs = pBandEngineGetNumOutputTensors(engine, model);
  printf("BandGetNumOutputTensors\n");

  int num_workers = pBandEngineGetNumWorkers(engine);
  printf("BandEngineGetNumWorkers\n");
  for (int i = 0; i < num_workers; i++) {
    printf("BandEngineGetWorkerDevice %d\n",
           pBandEngineGetWorkerDevice(engine, i));
  }

  BandTensor* input_tensor = pBandEngineCreateInputTensor(engine, model, 0);
  printf("BandEngineCreateInputTensor\n");
  BandTensor* output_tensor = pBandEngineCreateOutputTensor(engine, model, 0);
  printf("BandEngineCreateOutputTensor\n");

  int execution_count = 0;
  pBandEngineSetOnEndRequest(engine, on_end_request, &execution_count);
  printf("BandEngineSetOnEndRequest\n");

  float input[] = {1.f, 3.f};
  memcpy(pBandTensorGetData(input_tensor), input, 2 * sizeof(float));
  printf("BandTensorGetData\n");
  pBandTensorGetNumDims(input_tensor);
  printf("BandTensorNumDims\n");
  pBandEngineRequestSync(engine, model, &input_tensor, &output_tensor);
  printf("BandEngineRequestSync\n");

  if (execution_count != 1) {
    printf(
        "BandEngineSetOnEndRequest not worked in RequestSync (callback not "
        "called)\n");
  }

  BandRequestHandle request_handle =
      pBandEngineRequestAsync(engine, model, &input_tensor);
  printf("BandEngineRequestAsync\n");
  pBandEngineWait(engine, request_handle, &output_tensor, num_outputs);

  if (execution_count != 2) {
    printf(
        "BandEngineSetOnEndRequest not worked in RequestAsync (callback not "
        "called)\n");
  }

  BandRequestOption options;
  options.target_worker = -1;
  options.require_callback = false;

  request_handle =
      pBandEngineRequestAsyncOptions(engine, model, options, &input_tensor);
  printf("BandEngineRequestAsyncOptions\n");
  pBandEngineWait(engine, request_handle, &output_tensor, num_outputs);

  if (execution_count != 2) {
    printf(
        "BandEngineSetOnEndRequest should not be triggered with "
        "BandEngineRequestAsyncOptions\n");
  }

  request_handle = pBandEngineRequestSyncOptions(engine, model, options,
                                                 &input_tensor, &output_tensor);
  printf("BandEngineRequestSyncOptions\n");
  pBandEngineWait(engine, request_handle, &output_tensor, num_outputs);

  if (execution_count != 2) {
    printf(
        "BandEngineSetOnEndRequest should not be triggered with "
        "BandEngineRequestSyncOptions\n");
  }

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
