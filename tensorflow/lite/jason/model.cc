#include "tensorflow/lite/jason/model.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <unistd.h>

Model::Model(uint32_t c, uint32_t hw, int numConvs, std::string device, const NnApi* nnapi)
    : types(3*numConvs+1), nnapi(nnapi) {

  CHECK_NNAPI(nnapi->ANeuralNetworksModel_create(&model));
  devices.push_back(GetDevice(device));
  std::fill_n(supportedOps, 1000, false);

  paddingType.type = ANEURALNETWORKS_INT32;
  paddingType.dimensions = NULL;
  paddingType.dimensionCount = 0;
  paddingType.scale = 0.f;
  paddingType.zeroPoint = 0;

  strideWidthType.type = ANEURALNETWORKS_INT32;
  strideWidthType.dimensions = NULL;
  strideWidthType.dimensionCount = 0;
  strideWidthType.scale = 0.f;
  strideWidthType.zeroPoint = 0;

  strideHeightType.type = ANEURALNETWORKS_INT32;
  strideHeightType.dimensions = NULL;
  strideHeightType.dimensionCount = 0;
  strideHeightType.scale = 0.f;
  strideHeightType.zeroPoint = 0;

  fuseType.type = ANEURALNETWORKS_INT32;
  fuseType.dimensions = NULL;
  fuseType.dimensionCount = 0;
  fuseType.scale = 0.f;
  fuseType.zeroPoint = 0;


  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &paddingType));      // operand 0
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &strideWidthType));  // operand 1
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &strideHeightType)); // operand 2
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &fuseType));         // operand 3

  // int32_t noneValue = ANEURALNETWORKS_FUSED_NONE;
  int32_t paddingValue = ANEURALNETWORKS_PADDING_VALID;
  // int32_t paddingValue = ANEURALNETWORKS_PADDING_SAME;
  int32_t strideWidthValue = 1;
  int32_t strideHeightValue = 1;
  int32_t fuseValue = ANEURALNETWORKS_FUSED_RELU6;
  // int32_t fuseValue = ANEURALNETWORKS_FUSED_NONE;
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 0, &paddingValue, sizeof(paddingValue)));
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 1, &strideWidthValue, sizeof(strideWidthValue)));
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 2, &strideHeightType, sizeof(strideHeightValue)));
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 3, &fuseValue, sizeof(fuseValue)));


  uint32_t inputOutputDims[4] = {1, hw, hw, c};
  uint32_t filterDims[4] = {c, 1, 1, c};
  uint32_t biasDims[1] = {c};
  float r1 = static_cast<float>(rand()) / (static_cast <float>(RAND_MAX)) / 2 + 1.0f;
  int r2 = rand() % 5;

  ANeuralNetworksOperandType& realInputType = types[0];
  realInputType.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
  realInputType.dimensions = inputOutputDims;
  realInputType.dimensionCount = 4;
  realInputType.scale = r1;
  realInputType.zeroPoint = r2;

  CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &realInputType));         // operand 4


  for (int i = 0; i < numConvs; ++i) {
    float r3 = static_cast<float>(rand()) / (static_cast <float>(RAND_MAX)) / 2 + 1.0f;
    int r4 = rand() % 5;

    ANeuralNetworksOperandType& filterType = types[3*i + 1];
    filterType.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
    filterType.dimensions = filterDims;
    filterType.dimensionCount = 4;
    filterType.scale = r3;
    filterType.zeroPoint = r4;

    ANeuralNetworksOperandType& biasType = types[3*i + 2];
    biasType.type = ANEURALNETWORKS_TENSOR_INT32;
    biasType.dimensions = biasDims;
    biasType.dimensionCount = 1;
    biasType.scale = r1 * r3;
    biasType.zeroPoint = 0;

    r1 = r1 * r3 + 0.1f;

    ANeuralNetworksOperandType& outputType = types[3*i + 3];
    outputType.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
    outputType.dimensions = inputOutputDims;
    outputType.dimensionCount = 4;
    outputType.scale = r1;
    outputType.zeroPoint = r2;


    CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &filterType));         // operand 3*i+5
    CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &biasType));           // operand 3*i+6
    CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperand(model, &outputType));         // operand 3*i+7

    
    filterValuesVec.emplace_back();
    std::vector<uint8_t>& filterValues = filterValuesVec.back();
    for (int i = 0; i < c*1*1*c; ++i) {
      filterValues.push_back(rand() % 64);
    }
    
    biasValuesVec.emplace_back();
    std::vector<int>& biasValues = biasValuesVec.back();
    for (int i = 0; i < c; ++i) {
      biasValues.push_back(rand() % 256);
    }

    CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 3*i+5, filterValues.data(), sizeof(uint8_t) * filterValues.size()));
    CHECK_NNAPI(nnapi->ANeuralNetworksModel_setOperandValue(model, 3*i+6, biasValues.data(), sizeof(int) * biasValues.size()));


    convInIndicesVec.emplace_back();
    std::vector<uint32_t>& convInIndices = convInIndicesVec.back();
    convOutIndicesVec.emplace_back();
    std::vector<uint32_t>& convOutIndices = convOutIndicesVec.back();
    convInIndices.push_back(3*i+4); // input
    convInIndices.push_back(3*i+5); // filter
    convInIndices.push_back(3*i+6); // bias
    convInIndices.push_back(0);     // padding
    convInIndices.push_back(1);     // strideWidth
    convInIndices.push_back(2);     // strideHeight
    convInIndices.push_back(3);     // fuse
    convOutIndices.push_back(3*i+7);

    CHECK_NNAPI(nnapi->ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 7, convInIndices.data(), 1, convOutIndices.data()));
  }

  modelInputIndexes = 4;
  modelOutputIndexes = 3 * numConvs + 4;
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, &modelInputIndexes, 1, &modelOutputIndexes));
  CHECK_NNAPI(nnapi->ANeuralNetworksModel_finish(model));


  CHECK_NNAPI(nnapi->ANeuralNetworksModel_getSupportedOperationsForDevices(model, &devices[0], devices.size(), supportedOps));
  for (int i = 0; i < numConvs; ++i) {
    assert(supportedOps[i]);
  }

  CHECK_NNAPI(nnapi->ANeuralNetworksCompilation_createForDevices(model, &devices[0], devices.size(), &compilation));
  // nnapi->ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_LOW_POWER);
  // ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER
  // ANEURALNETWORKS_PREFER_SUSTAINED_SPEED
  CHECK_NNAPI(nnapi->ANeuralNetworksCompilation_finish(compilation));
}


Model::~Model() {
  nnapi->ANeuralNetworksCompilation_free(compilation);
  nnapi->ANeuralNetworksModel_free(model);
}


void Model::FinishCompilation() {
}


ANeuralNetworksDevice* Model::GetDevice(std::string deviceName) {
  uint32_t numDevices = 0;
  CHECK_NNAPI(nnapi->ANeuralNetworks_getDeviceCount(&numDevices));

  for (uint32_t i = 0; i < numDevices; i++) {
    ANeuralNetworksDevice* device = nullptr;
    const char* buffer = nullptr;
    CHECK_NNAPI(nnapi->ANeuralNetworks_getDevice(i, &device));
    CHECK_NNAPI(nnapi->ANeuralNetworksDevice_getName(device, &buffer));

    if (deviceName == buffer) {
      return device;
    }
  }

  std::cout << "Could not find " << deviceName << std::endl;
  exit(EXIT_FAILURE);
}

