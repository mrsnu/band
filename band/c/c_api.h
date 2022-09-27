#ifndef BAND_C_C_API_H_
#define BAND_C_C_API_H_

#include <stdarg.h>
#include <stdint.h>

#include "common.h"

#ifdef SWIG
#define BAND_CAPI_EXPORT
#else
// TODO: Add BAND_CAPI_EXPORT flag to support external symbols to dll (windows
// platform)
#if defined(_WIN32)
#ifdef BAND_COMPILE_LIBRARY
#define BAND_CAPI_EXPORT __declspec(dllexport)
#else
#define BAND_CAPI_EXPORT __declspec(dllimport)
#endif  // BAND_COMPILE_LIBRARY
#else
#define BAND_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Forward decl of internal types - details are in `c_api_type.h`
typedef struct BandConfig BandConfig;
typedef struct BandModel BandModel;
typedef struct BandTensor BandTensor;
typedef struct BandEngine BandEngine;
typedef struct BandRequestHandle BandRequestHandle;

/* config */
BAND_CAPI_EXPORT extern BandConfig* BandConfigCreate(const void* config_data,
                                                     size_t config_size);
BAND_CAPI_EXPORT extern BandConfig* BandConfigCreateFromFile(
    const char* config_path);
BAND_CAPI_EXPORT extern void BandConfigDelete(BandConfig* config);

/* model */
BAND_CAPI_EXPORT extern BandModel* BandModelCreate();
BAND_CAPI_EXPORT extern void BandModelDelete(BandModel* model);
BAND_CAPI_EXPORT extern BandStatus BandModelAddFromBuffer(
    BandModel* model, BandBackendType backend_type, const void* model_data,
    size_t model_size);
BAND_CAPI_EXPORT extern BandStatus BandModelAddFromFile(
    BandModel* model, BandBackendType backend_type, const char* model_path);

/* tensor */
// Band intetionally `only` expose getters to ensure
BAND_CAPI_EXPORT extern void BandTensorDelete(BandTensor* tensor);
BAND_CAPI_EXPORT extern BandType BandTensorGetType(BandTensor* tensor);
BAND_CAPI_EXPORT extern void* BandTensorGetData(BandTensor* tensor);
BAND_CAPI_EXPORT extern int* BandTensorGetDims(BandTensor* tensor);
BAND_CAPI_EXPORT extern size_t BandTensorGetBytes(BandTensor* tensor);
BAND_CAPI_EXPORT extern const char* BandTensorGetName(BandTensor* tensor);
BAND_CAPI_EXPORT extern BandQuantization BandTensorGetQuantization(
    BandTensor* tensor);

/* engine */
// TODO: Error reporter
BAND_CAPI_EXPORT extern BandEngine* BandEngineCreate(BandConfig* config);
BAND_CAPI_EXPORT extern void BandEngineDelete(BandEngine* engine);
BAND_CAPI_EXPORT extern BandStatus BandEngineRegisterModel(BandEngine* engine,
                                                           BandModel* model);
BAND_CAPI_EXPORT extern int BandEngineGetNumInputTensors(BandEngine* engine,
                                                         BandModel* model);
BAND_CAPI_EXPORT extern int BandEngineGetNumOutputTensors(BandEngine* engine,
                                                          BandModel* model);
// Create a input tensor for given model's n'th index
BAND_CAPI_EXPORT extern BandTensor* BandEngineCreateInputTensor(
    BandEngine* engine, BandModel* model, int index);
// Create a output tensor for given model's n'th index
BAND_CAPI_EXPORT extern BandTensor* BandEngineCreateOutputTensor(
    BandEngine* engine, BandModel* model, int index);
BAND_CAPI_EXPORT extern BandStatus BandEngineRequestSync(
    BandEngine* engine, BandModel* model, BandTensor** input_tensors,
    BandTensor** output_tensors);
BAND_CAPI_EXPORT extern BandRequestHandle BandEngineRequestAsync(
    BandEngine* engine, BandModel* model, BandTensor** input_tensors);
BAND_CAPI_EXPORT extern BandStatus BandEngineWait(BandEngine* engine,
                                                  BandRequestHandle handle,
                                                  BandTensor** output_tensors);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif
