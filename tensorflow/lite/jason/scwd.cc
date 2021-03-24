#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <chrono>
#include <sys/mman.h>
#include <algorithm>

#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"

#define IN_C 32
#define OUT_C 32
#define HW 56
#define NUM_CONVS 10


using namespace std::chrono;


ANeuralNetworksDevice* GetDevice(const NnApi* nnapi, std::string device_name) {
  uint32_t num_devices = 0;
  CHECK_NNAPI(nnapi->ANeuralNetworks_getDeviceCount(&num_devices));

  for (uint32_t i = 0; i < num_devices; i++) {
    ANeuralNetworksDevice* device = nullptr;
    const char* buffer = nullptr;
    CHECK_NNAPI(nnapi->ANeuralNetworks_getDevice(i, &device));
    CHECK_NNAPI(nnapi->ANeuralNetworksDevice_getName(device, &buffer));

    if (device_name == buffer) {
      return device;
    }
  }

  std::cout << "Could not find " << device_name << std::endl;
  exit(EXIT_FAILURE);
}



int main() {
  std::cout << "Yesyes" << std::endl;


  const NnApi* nnapi = NnApiImplementation();

  ANeuralNetworksDevice* device = GetDevice(nnapi, "qti-dsp");
  std::vector<ANeuralNetworksDevice*> devices;
  devices.push_back(device);
  bool supportedOps[N];
  std::fill_n(supportedOps, N, false);


  CHECK_NNAPI(nnapi->ANeuralNetworksModel_getSupportedOperationsForDevices(model, &devices[0], devices.size(), supportedOps));
  std::cout << "Supported: " << supportedOps[0] << std::endl;



  ANeuralNetworksCompilation* compilation;
  // CHECK_NNAPI(nnapi->ANeuralNetworksCompilation_create(model, &compilation));
  CHECK_NNAPI(nnapi->ANeuralNetworksCompilation_createForDevices(model, &devices[0], devices.size(), &compilation));


  // nnapi->ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_LOW_POWER);
  // ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER
  // ANEURALNETWORKS_PREFER_SUSTAINED_SPEED

  CHECK_NNAPI(nnapi->ANeuralNetworksCompilation_finish(compilation));




  

  // float myInput[3][4] = {{1.3, 1.4, 1.5, 1.6},
                           // {1.7, 1.8, 1.9, 2.0},
                           // {2.1, 2.2, 2.3, 2.4}};
  uint8_t myInput[1][HW][HW][IN_C];
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < HW; ++j) {
      for (int k = 0; k < HW; ++k) {
        for (int l = 0; l < IN_C; ++l) {
          myInput[i][j][k][l] = rand() % 64;
        }
      }
    }
  }
  uint8_t myOutput[1][HW][HW][OUT_C];

  // int t12s[1000];
  // int t23s[1000];
  ANeuralNetworksExecution* execs[1000];
  ANeuralNetworksEvent* events[1000];
  for (int i = 0; i < 1000; ++i) {
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_create(compilation, &execs[i]));
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setInput(execs[i], 0, NULL, myInput, sizeof(myInput)));
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setOutput(execs[i], 0, NULL, myOutput, sizeof(myOutput)));
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setMeasureTiming(execs[i], true));
  }


  for (int i = 0; i < 10; ++i) {
    ANeuralNetworksExecution* execWarmup = NULL;
    ANeuralNetworksEvent* eventWarmup = NULL;
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_create(compilation, &execWarmup));
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setInput(execWarmup, 0, NULL, myInput, sizeof(myInput)));
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setOutput(execWarmup, 0, NULL, myOutput, sizeof(myOutput)));
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_startCompute(execWarmup, &eventWarmup));
    nnapi->ANeuralNetworksEvent_free(eventWarmup);
    nnapi->ANeuralNetworksExecution_free(execWarmup);
  }


  auto start = high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    if (i == 0) {
      CHECK_NNAPI(nnapi->ANeuralNetworksExecution_startCompute(execs[i], &events[i]));
    } else {
      CHECK_NNAPI(nnapi->ANeuralNetworksExecution_startComputeWithDependencies(execs[i], &events[i-1], 1, 0, &events[i]));  
    }
    // nnapi->ANeuralNetworksEvent_free(run_end);
    // nnapi->ANeuralNetworksExecution_free(run);
  }

  for (int i = 0; i < 1000; ++i) {
    CHECK_NNAPI(nnapi->ANeuralNetworksEvent_wait(events[i]));
  }
  // CHECK_NNAPI(nnapi->ANeuralNetworksEvent_wait(events[999]));
  auto stop = high_resolution_clock::now();

  std::cout << duration_cast<microseconds>(stop - start).count() << " us" << std::endl;

  uint64_t onHardware = 0, inDriver = 0;
  // uint64_t fencedHardware = 0, fencedDriver = 0;
  uint64_t curr  = 0;
  for (int i = 0; i < 1000; ++i) {
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_getDuration(execs[i], ANEURALNETWORKS_DURATION_ON_HARDWARE, &curr));
    onHardware += curr;
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_getDuration(execs[i], ANEURALNETWORKS_DURATION_IN_DRIVER, &curr));
    inDriver += curr;
    // CHECK_NNAPI(nnapi->ANeuralNetworksExecution_getDuration(execs[i], ANEURALNETWORKS_FENCED_DURATION_ON_HARDWARE, &curr));
    // fencedHardware += curr;
    // CHECK_NNAPI(nnapi->ANeuralNetworksExecution_getDuration(execs[i], ANEURALNETWORKS_FENCED_DURATION_IN_DRIVER, &curr));
    // fencedDriver += curr;
  }
  std::cout <<  "ANEURALNETWORKS_DURATION_ON_HARDWARE (ns): " << onHardware << std::endl;
  std::cout <<  "ANEURALNETWORKS_DURATION_IN_DRIVER (ns): " << inDriver << std::endl;
  // std::cout <<  "ANEURALNETWORKS_FENCED_DURATION_ON_HARDWARE (ns): " << fencedHardware << std::endl;
  // std::cout <<  "ANEURALNETWORKS_FENCED_DURATION_IN_DRIVER (ns): " << fencedDriver << std::endl;
  

  for (int i = 0; i < 1000; ++i) {
    nnapi->ANeuralNetworksEvent_free(events[i]);
    nnapi->ANeuralNetworksExecution_free(execs[i]);
  }



  // for (int i = 0; i < 1000; ++i) {
  //   // std::cout << "Run " << i << std::endl;
  //   auto t1 = high_resolution_clock::now(); 
  //   ANeuralNetworksExecution* run = NULL;
  //   CHECK_NNAPI(nnapi->ANeuralNetworksExecution_create(compilation, &run));
  //   CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setInput(run, 0, NULL, myInput, sizeof(myInput)));
  //   CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setOutput(run, 0, NULL, myOutput, sizeof(myOutput)));
  //   ANeuralNetworksEvent* run_end = NULL;

  //   auto t2 = high_resolution_clock::now();
    
  //   CHECK_NNAPI(nnapi->ANeuralNetworksExecution_startCompute(run, &run_end));
    
  //   auto t3 = high_resolution_clock::now();
    
  //   t12s[i] = duration_cast<microseconds>(t2 - t1).count();
  //   t23s[i] = duration_cast<microseconds>(t3 - t2).count();
  //   nnapi->ANeuralNetworksEvent_free(run_end);
  //   nnapi->ANeuralNetworksExecution_free(run);
  // }
  
  // std::cout << t12s[0]   << " " << t23s[0]   << " us" << std::endl;
  // std::cout << t12s[1]   << " " << t23s[1]   << " us" << std::endl;
  // std::cout << t12s[2]   << " " << t23s[2]   << " us" << std::endl;
  // std::cout << t12s[3]   << " " << t23s[3]   << " us" << std::endl;
  // std::cout << t12s[997] << " " << t23s[997] << " us" << std::endl;
  // std::cout << t12s[998] << " " << t23s[998] << " us" << std::endl;
  // std::cout << t12s[999] << " " << t23s[999] << " us" << std::endl;

  
  
  nnapi->ANeuralNetworksCompilation_free(compilation);
  // nnapi->ANeuralNetworksMemory_free(mem);
  nnapi->ANeuralNetworksModel_free(model);

  std::cout << "Byebye" << std::endl;
}
