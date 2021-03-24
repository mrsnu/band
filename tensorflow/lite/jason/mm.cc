#include <string>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"



#define CHECK_NNAPI(...)                                                      \
  do {                                                                        \
    int status = __VA_ARGS__;                                                 \
    if (status != ANEURALNETWORKS_NO_ERROR) {                                 \
      printf("NNAPI fail at %s:%d '%s' with error: %d\n", __FILE__, __LINE__, \
             #__VA_ARGS__, status);                                           \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)


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

  ANeuralNetworksDevice* device = GetDevice(nnapi, "google-edgetpu");
  std::vector<ANeuralNetworksDevice*> devices;
  devices.push_back(device);
  bool supportedOps[1] = {true};


  // ANeuralNetworksMemory* mem = NULL;
  // int fd = open("weights", O_RDONLY);
  // CHECK_NNAPI(nnapi->ANeuralNetworksMemory_createFromFd(sizeof(float) * 12, PROT_READ, fd, 0, &mem));
  // close(fd);
  


  ANeuralNetworksModel* model = NULL;
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_create(&model));


  // In our example, all our tensors are matrices of dimension [3][4]
  ANeuralNetworksOperandType tensor3x4Type;
  // tensor3x4Type.type = ANEURALNETWORKS_TENSOR_FLOAT32;
  tensor3x4Type.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
  tensor3x4Type.scale = 2.1f;    // These fields are used for quantized tensors
  tensor3x4Type.zeroPoint = 0;  // These fields are used for quantized tensors
  tensor3x4Type.dimensionCount = 2;
  uint32_t dims[2] = {3, 4};
  tensor3x4Type.dimensions = dims;

  ANeuralNetworksOperandType tensor3x4OutType;
  tensor3x4OutType.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
  tensor3x4OutType.scale = 10.2f;    // These fields are used for quantized tensors
  tensor3x4OutType.zeroPoint = 0;  // These fields are used for quantized tensors
  tensor3x4OutType.dimensionCount = 2;
  tensor3x4OutType.dimensions = dims;


  // We also specify operands that are activation function specifiers
  ANeuralNetworksOperandType activationType;
  activationType.type = ANEURALNETWORKS_INT32;
  activationType.scale = 0.f;
  activationType.zeroPoint = 0;
  activationType.dimensionCount = 0;
  activationType.dimensions = NULL;


  // Now we add the seven operands, in the same order defined in the diagram
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &tensor3x4Type));  // operand 0
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &tensor3x4Type));  // operand 1
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &activationType)); // operand 2
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &tensor3x4Type));  // operand 3
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &tensor3x4Type));  // operand 4
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &activationType)); // operand 5
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &tensor3x4OutType));  // operand 6

  uint8_t values[] = {1, 2, 3, 4,
                      5, 6, 7, 8,
                      9, 10, 11, 12};
  // In our example, operands 1 and 3 are constant tensors whose values were
  // established during the training process
  // const int sizeOfTensor = 3 * 4 * 4;    // The formula for size calculation is dim0 * dim1 * elementSize
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValueFromMemory(model, 1, mem, 0, sizeOfTensor));
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 1, values, sizeof(uint8_t) * 12));
  // CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValueFromMemory(model, 3, mem, 0, sizeOfTensor));
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 3, values, sizeof(uint8_t) * 12));


  // We set the values of the activation operands, in our example operands 2 and 5
  int32_t noneValue = ANEURALNETWORKS_FUSED_NONE;
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 2, &noneValue, sizeof(noneValue)));
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 5, &noneValue, sizeof(noneValue)));



  // We have two operations in our example
  // The first consumes operands 1, 0, 2, and produces operand 4
  uint32_t addInputIndexes[3] = {1, 0, 2};
  uint32_t addOutputIndexes[1] = {4};
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 3, addInputIndexes, 1, addOutputIndexes));


  // The second consumes operands 3, 4, 5, and produces operand 6
  uint32_t multInputIndexes[3] = {3, 4, 5};
  uint32_t multOutputIndexes[1] = {6};
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MUL, 3, multInputIndexes, 1, multOutputIndexes));

  // Our model has one input (0) and one output (6)
  uint32_t modelInputIndexes[1] = {0};
  uint32_t modelOutputIndexes[1] = {6};
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




  ANeuralNetworksExecution* run = NULL;
  CHECK_NNAPI(nnapi->ANeuralNetworksExecution_create(compilation, &run));

  // float myInput[3][4] = {{1.3, 1.4, 1.5, 1.6},
                           // {1.7, 1.8, 1.9, 2.0},
                           // {2.1, 2.2, 2.3, 2.4}};
  uint8_t myInput[3][4] = {{5, 4, 3, 2},
                         {1, 0, 1, 2},
                         {3, 4, 5, 6}};
  CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setInput(run, 0, NULL, myInput, sizeof(myInput)));

  uint8_t myOutput[3][4];
  CHECK_NNAPI(nnapi->ANeuralNetworksExecution_setOutput(run, 0, NULL, myOutput, sizeof(myOutput)));

  ANeuralNetworksEvent* run_end = NULL;
  CHECK_NNAPI(nnapi->ANeuralNetworksExecution_startCompute(run, &run_end));
  CHECK_NNAPI(nnapi->ANeuralNetworksEvent_wait(run_end));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      std::cout << (int)myOutput[i][j] << " ";
    }
    std::cout << std::endl;
  }
  


  nnapi->ANeuralNetworksEvent_free(run_end);
  nnapi->ANeuralNetworksExecution_free(run);
  nnapi->ANeuralNetworksCompilation_free(compilation);
  // nnapi->ANeuralNetworksMemory_free(mem);
  nnapi->ANeuralNetworksModel_free(model);

  std::cout << "Byebye" << std::endl;
}
