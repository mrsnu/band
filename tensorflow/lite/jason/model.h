#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"

#include <string>
#include <vector>

#define CHECK_NNAPI(...)                                                      \
  do {                                                                        \
    int status = __VA_ARGS__;                                                 \
    if (status != ANEURALNETWORKS_NO_ERROR) {                                 \
      printf("NNAPI fail at %s:%d '%s' with error: %d\n", __FILE__, __LINE__, \
             #__VA_ARGS__, status);                                           \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)



class Model {
 public:
  Model(uint32_t c, uint32_t hw, int numConvs, std::string device, const NnApi* nnapi);
  ~Model();

  void FinishCompilation();
  ANeuralNetworksDevice* GetDevice(std::string deviceName);

  const NnApi* nnapi;
  bool supportedOps[1000];

  std::vector<ANeuralNetworksOperandType> types;
  ANeuralNetworksOperandType paddingType;
  ANeuralNetworksOperandType strideWidthType;
  ANeuralNetworksOperandType strideHeightType;
  ANeuralNetworksOperandType fuseType;

  ANeuralNetworksModel* model;
  std::vector<ANeuralNetworksDevice*> devices;
  ANeuralNetworksCompilation* compilation;

  std::vector<std::vector<uint8_t>> filterValuesVec;
  std::vector<std::vector<int>> biasValuesVec;

  std::vector<std::vector<uint32_t>> convInIndicesVec;
  std::vector<std::vector<uint32_t>> convOutIndicesVec;

  uint32_t modelInputIndexes;
  uint32_t modelOutputIndexes;
};
