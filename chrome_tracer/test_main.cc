#include <string>

#include "chrome_tracer/tracer.h"

static const char* kAnalysisPipeline = "AnalysisPipeline";
static const char* kRenderingPipeline = "RenderingPipeline";

static const char* kEventAnalysis1 = "EventAnalysis1";
static const char* kEventAnalysis2 = "EventAnalysis2";

static const char* kEventRendering1 = "EventRendering1";
static const char* kEventRendering2 = "EventRendering2";

int main() {
  chrome_tracer::ChromeTracer tracer("test_tracer");

  tracer.AddStream(kAnalysisPipeline);
  tracer.AddStream(kRenderingPipeline);

  int32_t handle = tracer.BeginEvent(kAnalysisPipeline, kEventAnalysis1);
  tracer.EndEvent(kAnalysisPipeline, handle);
  handle = tracer.BeginEvent(kRenderingPipeline, kEventRendering1);
  tracer.EndEvent(kRenderingPipeline, handle);
  handle = tracer.BeginEvent(kAnalysisPipeline, kEventAnalysis2);
  tracer.EndEvent(kAnalysisPipeline, handle);
  handle = tracer.BeginEvent(kRenderingPipeline, kEventRendering2);
  tracer.EndEvent(kRenderingPipeline, handle);
  handle = tracer.BeginEvent(kAnalysisPipeline, kEventAnalysis2);
  tracer.EndEvent(kAnalysisPipeline, handle);

  if (tracer.Validate()) {
    printf("Validated!\n");
  } else {
    printf("Invalid!\n");
  }

  std::cout << tracer.Dump();
  tracer.Dump("test.json");

  return 0;
}
