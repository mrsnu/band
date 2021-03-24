#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <fcntl.h>
#include <chrono>
#include <sys/mman.h>
#include <unistd.h>

#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"

#define IN_C 32
#define OUT_C 16
#define HW 56

#define CHECK_NNAPI(...)                                                      \
  do {                                                                        \
    int status = __VA_ARGS__;                                                 \
    if (status != ANEURALNETWORKS_NO_ERROR) {                                 \
      printf("NNAPI fail at %s:%d '%s' with error: %d\n", __FILE__, __LINE__, \
             #__VA_ARGS__, status);                                           \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)


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

  // std::ofstream out;
  // out.open("weights", std::ios::out | std::ios::binary);

  // float f[] = {1.2, 1.1, 1.0, 0.9,
  //              0.8, 0.7, 0.6, 0.5,
  //              0.4, 0.3, 0.2, 0.1};
  // out.write(reinterpret_cast<const char*>(f), sizeof(float) * 12);
  // out.close();

  // for (auto& device : tflite::nnapi::GetDeviceNamesList()) {
  //   std::cout << device << std::endl;
  // }


  const NnApi* nnapi = NnApiImplementation();

  ANeuralNetworksDevice* device = GetDevice(nnapi, "qti-dsp");
  std::vector<ANeuralNetworksDevice*> devices;
  devices.push_back(device);
  bool supportedOps[1] = {true};


  // ANeuralNetworksMemory* mem = NULL;
  // int fd = open("weights", O_RDONLY);
  // CHECK_NNAPI(nnapi->ANeuralNetworksMemory_createFromFd(sizeof(float) * 12, PROT_READ, fd, 0, &mem));
  // close(fd);

  float r1 = static_cast<float>(rand()) / (static_cast <float>(RAND_MAX / 20));
  int r2 = rand() % 128;
  float r3 = static_cast<float>(rand()) / (static_cast <float>(RAND_MAX / 20));
  int r4 = rand() % 128;
  


  ANeuralNetworksModel* model = NULL;
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_create(&model));

  ANeuralNetworksOperandType inputType;
  inputType.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
  uint32_t inputDims[4] = {1, HW, HW, IN_C};
  inputType.dimensions = inputDims;
  inputType.dimensionCount = 4;
  inputType.scale = r1;
  inputType.zeroPoint = r2;

  ANeuralNetworksOperandType filterType;
  filterType.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
  uint32_t filterDims[4] = {OUT_C, 1, 1, IN_C};
  filterType.dimensions = filterDims;
  filterType.dimensionCount = 4;
  filterType.scale = r3;
  filterType.zeroPoint = r4;

  ANeuralNetworksOperandType biasType;
  biasType.type = ANEURALNETWORKS_TENSOR_INT32;
  uint32_t biasDims[1] = {OUT_C};
  biasType.dimensions = biasDims;
  biasType.dimensionCount = 1;
  biasType.scale = r1 * r3;
  biasType.zeroPoint = 0;

  ANeuralNetworksOperandType paddingType;
  paddingType.type = ANEURALNETWORKS_INT32;
  paddingType.dimensions = NULL;
  paddingType.dimensionCount = 0;
  paddingType.scale = 0.f;
  paddingType.zeroPoint = 0;

  ANeuralNetworksOperandType strideWidthType;
  strideWidthType.type = ANEURALNETWORKS_INT32;
  strideWidthType.dimensions = NULL;
  strideWidthType.dimensionCount = 0;
  strideWidthType.scale = 0.f;
  strideWidthType.zeroPoint = 0;

  ANeuralNetworksOperandType strideHeightType;
  strideHeightType.type = ANEURALNETWORKS_INT32;
  strideHeightType.dimensions = NULL;
  strideHeightType.dimensionCount = 0;
  strideHeightType.scale = 0.f;
  strideHeightType.zeroPoint = 0;

  ANeuralNetworksOperandType fuseType;
  fuseType.type = ANEURALNETWORKS_INT32;
  fuseType.dimensions = NULL;
  fuseType.dimensionCount = 0;
  fuseType.scale = 0.f;
  fuseType.zeroPoint = 0;

  ANeuralNetworksOperandType outputType;
  outputType.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
  uint32_t outputDims[4] = {1, HW, HW, OUT_C};
  outputType.dimensions = outputDims;
  outputType.dimensionCount = 4;
  outputType.scale = r1 * r3 * 2;
  outputType.zeroPoint = r2;



  // // In our example, all our tensors are matrices of dimension [3][4]
  // ANeuralNetworksOperandType tensor3x4Type;
  // // tensor3x4Type.type = ANEURALNETWORKS_TENSOR_FLOAT32;
  // tensor3x4Type.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
  // tensor3x4Type.scale = r1;    // These fields are used for quantized tensors
  // tensor3x4Type.zeroPoint = 0;  // These fields are used for quantized tensors
  // tensor3x4Type.dimensionCount = 2;
  // uint32_t dims[2] = {3, 4};
  // tensor3x4Type.dimensions = dims;

  // ANeuralNetworksOperandType tensor3x4OutType;
  // tensor3x4OutType.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
  // tensor3x4OutType.scale = r1 * r1 * 2;    // These fields are used for quantized tensors
  // tensor3x4OutType.zeroPoint = 0;  // These fields are used for quantized tensors
  // tensor3x4OutType.dimensionCount = 2;
  // tensor3x4OutType.dimensions = dims;


  // // We also specify operands that are activation function specifiers
  // ANeuralNetworksOperandType activationType;
  // activationType.type = ANEURALNETWORKS_INT32;
  // activationType.scale = 0.f;
  // activationType.zeroPoint = 0;
  // activationType.dimensionCount = 0;
  // activationType.dimensions = NULL;


  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &inputType));        // operand 0
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &filterType));       // operand 1
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &biasType));         // operand 2
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &paddingType));      // operand 3
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &strideWidthType));  // operand 4
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &strideHeightType)); // operand 5
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &fuseType));         // operand 6
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &outputType));       // operand 7

  // Now we add the seven operands, in the same order defined in the diagram
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &tensor3x4Type));  // operand 0
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &tensor3x4Type));  // operand 1
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &activationType)); // operand 2
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &tensor3x4Type));  // operand 3
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &tensor3x4Type));  // operand 4
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &activationType)); // operand 5
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &tensor3x4OutType));  // operand 6

  std::vector<uint8_t> filterValues;
  for (int i = 0; i < OUT_C*1*1*IN_C; ++i) {
    filterValues.push_back(rand() % 64);
  }

  std::vector<int> biasValues;
  for (int i = 0; i < OUT_C; ++i) {
    biasValues.push_back(rand() % 256);
  }
  // In our example, operands 1 and 3 are constant tensors whose values were
  // established during the training process
  // const int sizeOfTensor = 3 * 4 * 4;    // The formula for size calculation is dim0 * dim1 * elementSize
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValueFromMemory(model, 1, mem, 0, sizeOfTensor));
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 1, filterValues.data(), sizeof(uint8_t) * filterValues.size()));
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValueFromMemory(model, 3, mem, 0, sizeOfTensor));
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 2, biasValues.data(), sizeof(int) * biasValues.size()));


  // We set the values of the activation operands, in our example operands 2 and 5
  // int32_t noneValue = ANEURALNETWORKS_FUSED_NONE;
  // int32_t paddingValue = ANEURALNETWORKS_PADDING_VALID;
  int32_t paddingValue = ANEURALNETWORKS_PADDING_SAME;
  int32_t strideWidthValue = 1;
  int32_t strideHeightValue = 1;
  // int32_t fuseValue = ANEURALNETWORKS_FUSED_RELU6;
  int32_t fuseValue = ANEURALNETWORKS_FUSED_NONE;
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 3, &paddingValue, sizeof(paddingValue)));
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 4, &strideWidthValue, sizeof(strideWidthValue)));
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 5, &strideHeightType, sizeof(strideHeightValue)));
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 6, &fuseValue, sizeof(fuseValue)));

  uint32_t convInIndices[7] = {0, 1, 2, 3, 4, 5, 6};
  uint32_t convOutIndices[1] = {7};
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 7, convInIndices, 1, convOutIndices));



  // We have two operations in our example
  // The first consumes operands 1, 0, 2, and produces operand 4
  // uint32_t addInputIndexes[3] = {1, 0, 2};
  // uint32_t addOutputIndexes[1] = {4};
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 3, addInputIndexes, 1, addOutputIndexes));


  // The second consumes operands 3, 4, 5, and produces operand 6
  // uint32_t multInputIndexes[3] = {3, 4, 5};
  // uint32_t multOutputIndexes[1] = {6};
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MUL, 3, multInputIndexes, 1, multOutputIndexes));

  // Our model has one input (0) and one output (6)
  uint32_t modelInputIndexes[1] = {0};
  uint32_t modelOutputIndexes[1] = {7};
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, modelInputIndexes, 1, modelOutputIndexes));


  CHECK_NNAPI(nnapi->ANeuralNetworksModel_finish(model));
  



  CHECK_NNAPI(nnapi->ANeuralNetworksModel_getSupportedOperationsForDevices(model, &devices[0], devices.size(), supportedOps));
  std::cout << "Supported: " << supportedOps[0] << std::endl;



  ANeuralNetworksCompilation* compilation;
  // CHECK_NNAPI(nnapi->ANeuralNetworksCompilation_create(model, &compilation));
  CHECK_NNAPI(nnapi->ANeuralNetworksCompilation_createForDevices(model, &devices[0], devices.size(), &compilation));


  // nnapi->ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_LOW_POWER);
  // ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER
  // ANEURALNETWORKS_PREFER_SUSTAINED_SPEED

  CHECK_NNAPI(nnapi->ANeuralNetworksCompilation_finish(compilation));


  ANeuralNetworksBurst* burst;
  CHECK_NNAPI(nnapi->ANeuralNetworksBurst_create(compilation, &burst));




  

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
  for (int i = 0; i < 1000; ++i) {
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_create(compilation, &execs[i]));
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setInput(execs[i], 0, NULL, myInput, sizeof(myInput)));
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setOutput(execs[i], 0, NULL, myOutput, sizeof(myOutput)));
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
    CHECK_NNAPI(nnapi->ANeuralNetworksExecution_burstCompute(execs[i], burst));
  }
  auto stop = high_resolution_clock::now();

  std::cout << duration_cast<microseconds>(stop - start).count() << " us" << std::endl;

  for (int i = 0; i < 1000; ++i) {
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
  nnapi->ANeuralNetworksBurst_free(burst);
  nnapi->ANeuralNetworksModel_free(model);

  std::cout << "Byebye" << std::endl;
}
