#ifndef BAND_C_C_API_H_
#define BAND_C_C_API_H_

#include <stdarg.h>
#include <stdint.h>

#include "c_api_type.h"

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
typedef struct BandConfigBuilder BandConfigBuilder;
typedef struct BandConfig BandConfig;
typedef struct BandModel BandModel;
typedef struct BandTensor BandTensor;
typedef struct BandEngine BandEngine;
typedef int BandRequestHandle;

/* config builder */
BAND_CAPI_EXPORT extern BandConfigBuilder* BandConfigBuilderCreate();
BAND_CAPI_EXPORT extern void BandAddConfig(BandConfigBuilder* b, int field,
                                           int count, ...);
BAND_CAPI_EXPORT extern void BandConfigBuilderDelete(BandConfigBuilder* b);

/* config */
BAND_CAPI_EXPORT extern BandConfig* BandConfigCreate(BandConfigBuilder* b);
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
// Band intetionally `only` expose getters to ensure the tensor shape is
// immutable from C API.
BAND_CAPI_EXPORT extern void BandTensorDelete(BandTensor* tensor);
BAND_CAPI_EXPORT extern BandDataType BandTensorGetType(BandTensor* tensor);
BAND_CAPI_EXPORT extern void* BandTensorGetData(BandTensor* tensor);
BAND_CAPI_EXPORT extern size_t BandTensorGetNumDims(BandTensor* tensor);
BAND_CAPI_EXPORT extern const int* BandTensorGetDims(BandTensor* tensor);
BAND_CAPI_EXPORT extern size_t BandTensorGetBytes(BandTensor* tensor);
BAND_CAPI_EXPORT extern const char* BandTensorGetName(BandTensor* tensor);
BAND_CAPI_EXPORT extern BandQuantizationType BandTensorGetQuantizationType(
    BandTensor* tensor);
BAND_CAPI_EXPORT extern void* BandTensorGetQuantizationParams(
    BandTensor* tensor);

/* request option */
BAND_CAPI_EXPORT extern BandRequestOption BandRequestOptionGetDefault();

/* engine */
BAND_CAPI_EXPORT extern BandEngine* BandEngineCreate(BandConfig* config);
BAND_CAPI_EXPORT extern void BandEngineDelete(BandEngine* engine);
BAND_CAPI_EXPORT extern BandStatus BandEngineRegisterModel(BandEngine* engine,
                                                           BandModel* model);
BAND_CAPI_EXPORT extern int BandEngineGetNumInputTensors(BandEngine* engine,
                                                         BandModel* model);
BAND_CAPI_EXPORT extern int BandEngineGetNumOutputTensors(BandEngine* engine,
                                                          BandModel* model);

BAND_CAPI_EXPORT extern int BandEngineGetNumWorkers(BandEngine* engine);
BAND_CAPI_EXPORT extern BandDeviceFlag BandEngineGetWorkerDevice(
    BandEngine* engine, int worker_id);

// Create a input tensor for given model's n'th index
BAND_CAPI_EXPORT
extern BandTensor* BandEngineCreateInputTensor(BandEngine* engine,
                                               BandModel* model, size_t index);
// Create a output tensor for given model's n'th index
BAND_CAPI_EXPORT extern BandTensor* BandEngineCreateOutputTensor(
    BandEngine* engine, BandModel* model, size_t index);
BAND_CAPI_EXPORT extern BandStatus BandEngineRequestSync(
    BandEngine* engine, BandModel* model, BandTensor** input_tensors,
    BandTensor** output_tensors);
BAND_CAPI_EXPORT extern BandRequestHandle BandEngineRequestAsync(
    BandEngine* engine, BandModel* model, BandTensor** input_tensors);
BAND_CAPI_EXPORT extern BandStatus BandEngineRequestSyncOptions(
    BandEngine* engine, BandModel* model, BandRequestOption options,
    BandTensor** input_tensors, BandTensor** output_tensors);
BAND_CAPI_EXPORT extern BandRequestHandle BandEngineRequestAsyncOptions(
    BandEngine* engine, BandModel* model, BandRequestOption options,
    BandTensor** input_tensors);
BAND_CAPI_EXPORT extern BandStatus BandEngineWait(BandEngine* engine,
                                                  BandRequestHandle handle,
                                                  BandTensor** output_tensors,
                                                  size_t num_outputs);
BAND_CAPI_EXPORT extern void BandEngineSetOnEndRequest(
    BandEngine* engine,
    void (*on_end_invoke)(void* user_data, int job_id, BandStatus status),
    void* user_data);

typedef BandConfigBuilder* (*PFN_BandConfigBuilderCreate)();
typedef void (*PFN_BandAddConfig)(BandConfigBuilder*, int, int, ...);
typedef void (*PFN_BandConfigBuilderDelete)(BandConfigBuilder*);
typedef BandConfig* (*PFN_BandConfigCreate)(BandConfigBuilder*);
typedef void (*PFN_BandConfigDelete)(BandConfig*);
typedef BandModel* (*PFN_BandModelCreate)();
typedef void (*PFN_BandModelDelete)(BandModel*);
typedef BandStatus (*PFN_BandModelAddFromBuffer)(BandModel*, BandBackendType,
                                                 const void*, size_t);
typedef BandStatus (*PFN_BandModelAddFromFile)(BandModel*, BandBackendType,
                                               const char*);
typedef void (*PFN_BandTensorDelete)(BandTensor*);
typedef BandDataType (*PFN_BandTensorGetType)(BandTensor*);
typedef void* (*PFN_BandTensorGetData)(BandTensor*);
typedef size_t (*PFN_BandTensorGetNumDims)(BandTensor*);
typedef const int* (*PFN_BandTensorGetDims)(BandTensor*);
typedef size_t (*PFN_BandTensorGetBytes)(BandTensor*);
typedef const char* (*PFN_BandTensorGetName)(BandTensor*);
typedef BandQuantizationType (*PFN_BandTensorGetQuantizationType)(BandTensor*);
typedef void* (*PFN_BandTensorGetQuantizationParams)(BandTensor*);
typedef BandEngine* (*PFN_BandEngineCreate)(BandConfig*);
typedef void (*PFN_BandEngineDelete)(BandEngine*);
typedef BandStatus (*PFN_BandEngineRegisterModel)(BandEngine*, BandModel*);
typedef int (*PFN_BandEngineGetNumInputTensors)(BandEngine*, BandModel*);
typedef int (*PFN_BandEngineGetNumOutputTensors)(BandEngine*, BandModel*);
typedef int (*PFN_BandEngineGetNumWorkers)(BandEngine*);
typedef BandDeviceFlag (*PFN_BandEngineGetWorkerDevice)(BandEngine*, int);
typedef BandTensor* (*PFN_BandEngineCreateInputTensor)(BandEngine*, BandModel*,
                                                       size_t);
typedef BandTensor* (*PFN_BandEngineCreateOutputTensor)(BandEngine*, BandModel*,
                                                        size_t);
typedef BandStatus (*PFN_BandEngineRequestSync)(BandEngine*, BandModel*,
                                                BandTensor**, BandTensor**);
typedef BandRequestHandle (*PFN_BandEngineRequestAsync)(BandEngine*, BandModel*,
                                                        BandTensor**);
typedef BandStatus (*PFN_BandEngineRequestSyncOptions)(BandEngine*, BandModel*,
                                                       BandRequestOption,
                                                       BandTensor**,
                                                       BandTensor**);
typedef BandRequestHandle (*PFN_BandEngineRequestAsyncOptions)(
    BandEngine*, BandModel*, BandRequestOption, BandTensor**);
typedef BandStatus (*PFN_BandEngineWait)(BandEngine*, BandRequestHandle,
                                         BandTensor**, size_t);
typedef void (*PFN_BandEngineSetOnEndRequest)(BandEngine*,
                                              void (*)(void*, int, BandStatus),
                                              void*);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif
