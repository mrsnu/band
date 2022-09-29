#ifndef BAND_C_C_API_H_
#define BAND_C_C_API_H_

#include <stdarg.h>
#include <stdint.h>

#include "common.h"

#ifdef SWIG
#define BAND_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef BAND_COMPILE_LIBRARY
#define BAND_CAPI_EXPORT __declspec(dllexport)
#else
#define BAND_CAPI_EXPORT __declspec(dllimport)
#endif // BAND_COMPILE_LIBRARY
#else
#define BAND_CAPI_EXPORT __attribute__((visibility("default")))
#endif // _WIN32
#endif // SWIG

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
TODO
- Add functions
- Add representation for model / tensor in common
*/

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
#endif
