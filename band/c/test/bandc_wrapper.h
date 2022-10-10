#ifndef BAND_C_C_API_MAIN_H_
#define BAND_C_C_API_MAIN_H_

#include "band/c/c_api.h"

bool LoadBandLibrary();

typedef BandConfigBuilder*(*PFN_BandConfigBuilderCreate)();
typedef void(*PFN_BandAddConfig)(BandConfigBuilder*, int, int, ...);
typedef void(*PFN_BandConfigBuilderDelete)(BandConfigBuilder*);
typedef BandConfig*(*PFN_BandConfigCreate)(BandConfigBuilder*);
typedef void(*PFN_BandConfigDelete)(BandConfig*);
typedef BandModel*(*PFN_BandModelCreate)();
typedef void(*PFN_BandModelDelete)(BandModel*);
typedef BandStatus(*PFN_BandModelAddFromBuffer)(BandModel*, BandBackendType, const void*, size_t);
typedef BandStatus(*PFN_BandModelAddFromFile)(BandModel*, BandBackendType, const char*);
typedef void(*PFN_BandTensorDelete)(BandTensor*);
typedef BandType(*PFN_BandTensorGetType)(BandTensor*);
typedef void*(*PFN_BandTensorGetData)(BandTensor*);
typedef int*(*PFN_BandTensorGetDims)(BandTensor*);
typedef size_t(*PFN_BandTensorGetBytes)(BandTensor*);
typedef const char*(*PFN_BandTensorGetName)(BandTensor*);
typedef BandQuantization(*PFN_BandTensorGetQuantization)(BandTensor*);
typedef BandEngine*(*PFN_BandEngineCreate)(BandConfig*);
typedef void(*PFN_BandEngineDelete)(BandEngine*);
typedef BandStatus(*PFN_BandEngineRegisterModel)(BandEngine*, BandModel*);
typedef int(*PFN_BandEngineGetNumInputTensors)(BandEngine*, BandModel*);
typedef int(*PFN_BandEngineGetNumOutputTensors)(BandEngine*, BandModel*);
typedef BandTensor*(*PFN_BandEngineCreateInputTensor)(BandEngine*, BandModel*, size_t);
typedef BandTensor*(*PFN_BandEngineCreateOutputTensor)(BandEngine*, BandModel*, size_t);
typedef BandStatus(*PFN_BandEngineRequestSync)(BandEngine*, BandModel*, BandTensor**, BandTensor**);
typedef BandRequestHandle(*PFN_BandEngineRequestAsync)(BandEngine*, BandModel*, BandTensor**);
typedef BandStatus(*PFN_BandEngineWait)(BandEngine*, BandRequestHandle, BandTensor**, size_t);

extern PFN_BandConfigBuilderCreate BandConfigBuilderCreate;
extern PFN_BandAddConfig BandAddConfig;
extern PFN_BandConfigBuilderDelete BandConfigBuilderDelete;
extern PFN_BandConfigCreate BandConfigCreate;
extern PFN_BandConfigDelete BandConfigDelete;
extern PFN_BandModelCreate BandModelCreate;
extern PFN_BandModelDelete BandModelDelete;
extern PFN_BandModelAddFromBuffer BandModelAddFromBuffer;
extern PFN_BandModelAddFromFile BandModelAddFromFile;
extern PFN_BandTensorDelete BandTensorDelete;
extern PFN_BandTensorGetType BandTensorGetType;
extern PFN_BandTensorGetData BandTensorGetData;
extern PFN_BandTensorGetDims BandTensorGetDims;
extern PFN_BandTensorGetBytes BandTensorGetBytes;
extern PFN_BandTensorGetName BandTensorGetName;
extern PFN_BandTensorGetQuantization BandTensorGetQuantization;
extern PFN_BandEngineCreate BandEngineCreate;
extern PFN_BandEngineDelete BandEngineDelete;
extern PFN_BandEngineRegisterModel BandEngineRegisterModel;
extern PFN_BandEngineGetNumInputTensors BandEngineGetNumInputTensors;
extern PFN_BandEngineGetNumOutputTensors BandEngineGetNumOutputTensors;
extern PFN_BandEngineCreateInputTensor BandEngineCreateInputTensor;
extern PFN_BandEngineCreateOutputTensor BandEngineCreateOutputTensor;
extern PFN_BandEngineRequestSync BandEngineRequestSync;
extern PFN_BandEngineRequestAsync BandEngineRequestAsync;
extern PFN_BandEngineWait BandEngineWait;

#endif  // BAND_C_API_MAIN_H_