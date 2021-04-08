#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <unistd.h>

#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"
#include "tensorflow/lite/jason/model.h"

#define HW 56
#define C 32

using namespace std::chrono;

static int NUM_MODELS = 2;
static int NUM_CONVS = 100;
static std::string DEVICE = "google-edgetpu";
static bool ASYNC = false;


void printUsage() {
  std::cout << "Usage: ./test [-n N] [-c C] [-d D] [-s]"
            << std::endl;
  std::cout << "  -n: Number of models (default is 2)" << std::endl;
  std::cout << "  -c: Number of convs (default is 100)" << std::endl;
  std::cout << "  -d: Device to use (default is google-edgetpu, consider qti-dsp)" << std::endl;
  std::cout << "  -a: Async execution (ANeuralNetworksExecution_startCompute, default is off)" << std::endl;
}

int main(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:c:d:a")) != -1) {
    switch (c) {
      case 'n':
        NUM_MODELS = atoi(optarg);
        break;
      case 'c':
        NUM_CONVS = atoi(optarg);
        break;
      case 'd':
        DEVICE = optarg;
        break;
      case 'a':
        ASYNC = true;
        break;
      case '?':
        printUsage();
        return 1;
      default:
        break;
    }
  }

  if (optind < argc) {
    printUsage();
    return 1;
  }

  std::cout << DEVICE << " "
            << NUM_CONVS << " "
            << NUM_MODELS << " "
            << ASYNC << " "
            << std::endl;

  const NnApi* nnapi = NnApiImplementation();
  std::vector<Model> models;
  models.reserve(NUM_MODELS);
  for (int n = 0; n < NUM_MODELS; ++n) {
    models.emplace_back(C, HW, NUM_CONVS, DEVICE, nnapi);
  }

  uint8_t tensors[NUM_MODELS+1][1][HW][HW][C];
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < HW; ++j) {
      for (int k = 0; k < HW; ++k) {
        for (int l = 0; l < C; ++l) {
          tensors[0][i][j][k][l] = rand() % 64;
        }
      }
    }
  }


  // ANeuralNetworksMemoryDesc* desc1;
  // CHECK_NNAPI(nnapi->ANeuralNetworksMemoryDesc_create(&desc1));
  // uint32_t dims[] = {1, HW, HW, C};
  // CHECK_NNAPI(nnapi->ANeuralNetworksMemoryDesc_setDimensions(desc1, 4, dims));
  // CHECK_NNAPI(nnapi->ANeuralNetworksMemoryDesc_addOutputRole(desc1, model1.compilation, 0, 1.0));
  // CHECK_NNAPI(nnapi->ANeuralNetworksMemoryDesc_finish(desc1));


  // ANeuralNetworksMemory* opaqueMem1;
  // CHECK_NNAPI(nnapi->ANeuralNetworksMemory_createFromDesc(desc1, &opaqueMem1));


  ANeuralNetworksExecution* execs[NUM_MODELS][1000];
  ANeuralNetworksEvent* events[NUM_MODELS][1000];
  for (int i = 0; i < 1000; ++i) {
    for (int n = 0; n < NUM_MODELS; ++n) {
      CHECK_NNAPI(nnapi->ANeuralNetworksExecution_create(models[n].compilation, &execs[n][i]));
      CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setInput(execs[n][i], 0, NULL, tensors[n], sizeof(tensors[n])));
      CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setOutput(execs[n][i], 0, NULL, tensors[n+1], sizeof(tensors[n+1])));
      // CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setOutputFromMemory(execs1[i], 0, NULL, opaqueMem1, 0, 0));//sizeof(uint8_t)*HW*HW*C));
      CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setMeasureTiming(execs[n][i], true));
    }
  }

  for (int i = 0; i < 500; ++i) {
    for (int n = 0; n < NUM_MODELS; ++n) {
      CHECK_NNAPI(nnapi->ANeuralNetworksExecution_compute(execs[n][i]));
    }
  }

  uint64_t t12 = 0;
  uint64_t t23 = 0;
  uint64_t t45 = 0;
  uint64_t t67[2] = {0, 0};

  for (int i = 500; i < 1000; ++i) {
    auto t1 = high_resolution_clock::now();
    for (int n = 0; n < NUM_MODELS; ++n) {
      if (ASYNC) {
        if (n == 0) {
          auto t4 = high_resolution_clock::now();
          // CHECK_NNAPI(nnapi->ANeuralNetworksExecution_startCompute(execs[n][i], &events[n][i]));
          CHECK_NNAPI(nnapi->ANeuralNetworksExecution_startComputeWithDependencies(execs[n][i], nullptr, 0, 0, &events[n][i]));
          auto t5 = high_resolution_clock::now();
          t45 += duration_cast<microseconds>(t5 - t4).count();
        } else {
          if (n == 1 && i == 500) {
            int fenceId;
            CHECK_NNAPI(nnapi->ANeuralNetworksEvent_getSyncFenceFd(events[n-1][i], &fenceId));
            std::cout << "Fence id: " << fenceId << std::endl;
          }


          auto t6 = high_resolution_clock::now();
          CHECK_NNAPI(nnapi->ANeuralNetworksExecution_startComputeWithDependencies(execs[n][i], &events[n-1][i], 1, 0, &events[n][i]));
          auto t7 = high_resolution_clock::now();
          t67[n-1] += duration_cast<microseconds>(t7 - t6).count();
        }
        
      } else {
        CHECK_NNAPI(nnapi->ANeuralNetworksExecution_compute(execs[n][i]));
      }
    }
    auto t2 = high_resolution_clock::now();

    if (ASYNC) {
      for (int n = 0; n < NUM_MODELS; ++n) {
        CHECK_NNAPI(nnapi->ANeuralNetworksEvent_wait(events[n][i]));
      }
    }
    auto t3 = high_resolution_clock::now();

    t12 += duration_cast<microseconds>(t2 - t1).count();
    t23 += duration_cast<microseconds>(t3 - t2).count();
  }

  std::cout << t12 + t23 << " us" << std::endl;
  std::cout << t12 << " us" << std::endl;
  std::cout << t23 << " us" << std::endl;
  std::cout << t45 << " us" << std::endl;
  std::cout << t67[0] << " us" << std::endl;
  std::cout << t67[1] << " us" << std::endl;

  uint64_t curr = 0;
  uint64_t onHardware[NUM_MODELS], inDriver[NUM_MODELS], onHardwareFenced[NUM_MODELS] , inDriverFenced[NUM_MODELS];
  for (int n = 0; n < NUM_MODELS; ++n) {
    onHardware[n] = 0;
    inDriver[n] = 0;
    onHardwareFenced[n] = 0;
    inDriverFenced[n] = 0;
  }

  for (int i = 500; i < 1000; ++i) {
    for (int n = 0; n < NUM_MODELS; ++n) {
      CHECK_NNAPI(nnapi->ANeuralNetworksExecution_getDuration(execs[n][i], ANEURALNETWORKS_DURATION_ON_HARDWARE, &curr));
      onHardware[n] += curr;
      CHECK_NNAPI(nnapi->ANeuralNetworksExecution_getDuration(execs[n][i], ANEURALNETWORKS_DURATION_IN_DRIVER, &curr));
      inDriver[n] += curr;
      CHECK_NNAPI(nnapi->ANeuralNetworksExecution_getDuration(execs[n][i], ANEURALNETWORKS_FENCED_DURATION_ON_HARDWARE, &curr));
      onHardwareFenced[n] += curr;
      CHECK_NNAPI(nnapi->ANeuralNetworksExecution_getDuration(execs[n][i], ANEURALNETWORKS_FENCED_DURATION_IN_DRIVER, &curr));
      inDriverFenced[n] += curr;
    }
  }

  for (int n = 0; n < NUM_MODELS; ++n) {
    std::cout <<  "ANEURALNETWORKS_DURATION_ON_HARDWARE        " << n << " (ns): " << onHardware[n] << std::endl;
  }
  for (int n = 0; n < NUM_MODELS; ++n) {
    std::cout <<  "ANEURALNETWORKS_DURATION_IN_DRIVER          " << n << " (ns): " << inDriver[n] << std::endl;
  }
  // for (int n = 0; n < NUM_MODELS; ++n) {
  //   std::cout <<  "ANEURALNETWORKS_FENCED_DURATION_ON_HARDWARE " << n << " (ns): " << onHardwareFenced[n] << std::endl;
  // }
  // for (int n = 0; n < NUM_MODELS; ++n) {
  //   std::cout <<  "ANEURALNETWORKS_FENCED_DURATION_IN_DRIVER   " << n << " (ns): " << inDriverFenced[n] << std::endl;
  // }

  // nnapi->ANeuralNetworksMemory_free(opaqueMem1);
  // nnapi->ANeuralNetworksMemoryDesc_free(desc1);
  for (int i = 0; i < 1000; ++i) {
    for (int n = 0; n < NUM_MODELS; ++n) {
      nnapi->ANeuralNetworksExecution_free(execs[n][i]);
    }  
  }
}
