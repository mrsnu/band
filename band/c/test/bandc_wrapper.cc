#include "band/c/c_api.h"

#if defined(_WIN32)
#define __WINDOWS__
#endif

#include <stdio.h>
#include <stdbool.h>

#ifdef __WINDOWS__
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef __WINDOWS__
#define LoadFunction(function) \
  function = (PFN_##function) GetProcAddress(libbandc, #function);
#else
#define LoadFunction(function) \
  function = (PFN_##function) dlsym(libbandc, #function);
#endif

#ifdef __WINDOWS__
void LoadBandLibraryFunctions(HMODULE libbandc) {
#else
void LoadBandLibraryFunctions(void* libbandc) {
  LoadFunction(BandAddConfig);
  LoadFunction(BandBackendGetName);
  LoadFunction(BandBackendGetType);
  LoadFunction(BandConfigBuilderCreate);
  LoadFunction(BandConfigBuilderDelete);
  LoadFunction(BandConfigCreate);
  LoadFunction(BandConfigDelete);
  LoadFunction(BandDeviceGetFlag);
  LoadFunction(BandDeviceGetName);
  LoadFunction(BandEngineCreate);
  LoadFunction(BandEngineCreateInputTensor);
  LoadFunction(BandEngineCreateOutputTensor);
  LoadFunction(BandEngineDelete);
  LoadFunction(BandEngineGetNumInputTensors);
  LoadFunction(BandEngineGetNumOutputTensors);
  LoadFunction(BandEngineRegisterModel);
  LoadFunction(BandEngineRequestAsync);
  LoadFunction(BandEngineRequestSync);
  LoadFunction(BandEngineWait);
  LoadFunction(BandFloatArrayCreate);
  LoadFunction(BandFloatArrayFree);
  LoadFunction(BandFloatArrayGetSizeInBytes);
  LoadFunction(BandIntArrayCopy);
  LoadFunction(BandIntArrayCreate);
  LoadFunction(BandIntArrayEqual);
  LoadFunction(BandIntArrayEqualsArray);
  LoadFunction(BandIntArrayFree);
  LoadFunction(BandIntArrayGetSizeInBytes);
  LoadFunction(BandModelAddFromBuffer);
  LoadFunction(BandModelAddFromFile);
  LoadFunction(BandModelCreate);
  LoadFunction(BandModelDelete);
  LoadFunction(BandTensorDelete);
  LoadFunction(BandTensorGetBytes);
  LoadFunction(BandTensorGetData);
  LoadFunction(BandTensorGetDims);
  LoadFunction(BandTensorGetName);
  LoadFunction(BandTensorGetQuantization);
  LoadFunction(BandTensorGetType);
  LoadFunction(BandTypeGetName);
}
#endif

bool LoadBandLibrary() {
#ifdef __WINDOWS__
  HMODULE libbandc = LoadLibraryA("band_c.dll");
  if (libbandc) {
    // Load functions
    return true;
  } else {
    DWORD error_code = GetLastError();
    fprintf(stderr, "Cannnot open Band C Library oon this device, error code - %d", error_code);
    return false;
  }
#else
  void *libbandc = NULL;
  libbandc = dlopen("libband_c.so", RTLD_NOW | RTLD_LOCAL);
  if (libbandc) {
    // Load functions
    return true;
  }
#endif
}
