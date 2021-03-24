#include <string>
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

int main(int argc, char** argv) {
  int N = atoi(argv[1]);

  const NnApi* nnapi = NnApiImplementation();
  Model model1(C, HW, N, "qti-dsp", nnapi);


  uint8_t myInput[1][HW][HW][C];
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < HW; ++j) {
      for (int k = 0; k < HW; ++k) {
        for (int l = 0; l < C; ++l) {
          myInput[i][j][k][l] = rand() % 64;
        }
      }
    }
  }
  uint8_t myHidden[1][HW][HW][C];
  uint8_t myOutput[1][HW][HW][C];


  // ANeuralNetworksMemoryDesc* desc1;
  // CHECK_NNAPI(nnapi->ANeuralNetworksMemoryDesc_create(&desc1));
  // uint32_t dims[] = {1, HW, HW, C};
  // CHECK_NNAPI(nnapi->ANeuralNetworksMemoryDesc_setDimensions(desc1, 4, dims));
  // CHECK_NNAPI(nnapi->ANeuralNetworksMemoryDesc_addOutputRole(desc1, model1.compilation, 0, 1.0));
  // CHECK_NNAPI(nnapi->ANeuralNetworksMemoryDesc_finish(desc1));


  // ANeuralNetworksMemory* opaqueMem1;
  // CHECK_NNAPI(nnapi->ANeuralNetworksMemory_createFromDesc(desc1, &opaqueMem1));


  ANeuralNetworksExecution* execs1[1000];
  for (int i = 0; i < 1000; ++i) {
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_create(model1.compilation, &execs1[i]));
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setInput(execs1[i], 0, NULL, myInput, sizeof(myInput)));
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setOutput(execs1[i], 0, NULL, myHidden, sizeof(myHidden)));
    // CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setOutputFromMemory(execs1[i], 0, NULL, opaqueMem1, 0, 0));//sizeof(uint8_t)*HW*HW*C));
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setMeasureTiming(execs1[i], true));
  }

  for (int i = 0; i < 500; ++i) {
    // std::cout << i << std::endl;
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_compute(execs1[i]));
  }
  auto start = high_resolution_clock::now();
  for (int i = 500; i < 1000; ++i) {
    // std::cout << i << std::endl;
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_compute(execs1[i]));
  }
  auto stop = high_resolution_clock::now();

  std::cout << duration_cast<microseconds>(stop - start).count() << " us" << std::endl;

  uint64_t onHardware = 0, inDriver = 0;
  uint64_t curr  = 0;
  for (int i = 500; i < 1000; ++i) {
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_getDuration(execs1[i], ANEURALNETWORKS_DURATION_ON_HARDWARE, &curr));
    onHardware += curr;
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_getDuration(execs1[i], ANEURALNETWORKS_DURATION_IN_DRIVER, &curr));
    inDriver += curr;
  }
  std::cout <<  "ANEURALNETWORKS_DURATION_ON_HARDWARE (ns): " << onHardware << std::endl;
  std::cout <<  "ANEURALNETWORKS_DURATION_IN_DRIVER (ns): " << inDriver << std::endl;

  // nnapi->ANeuralNetworksMemory_free(opaqueMem1);
  // nnapi->ANeuralNetworksMemoryDesc_free(desc1);
  for (int i = 0; i < 1000; ++i) {
    nnapi->ANeuralNetworksExecution_free(execs1[i]);
  }
}
