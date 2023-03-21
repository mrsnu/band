#ifndef BAND_TRACER_H_
#define BAND_TRACER_H_

#include "chrome_tracer/tracer.h"

namespace band {

class Tracer {};
}  // namespace band

#ifdef BAND_TRACE
BAND_TRACER_ADD_STREAM(name)

#elif
#define BAND_TRACER_ADD_STREAM(name) ...

#endif