#include "tensorflow/lite/tools/logging_reporter.h"
#include "tensorflow/lite/tools/logging.h"

#include <memory>

namespace tflite {
int LoggingReporter::Report(const char* format, va_list args) {
  int count = snprintf(nullptr, 0, format, args) + 1;
  char* buffer = new char[count];
  vsnprintf(buffer, count, format, args);
  TFLITE_LOG(ERROR) << buffer;
  delete[] buffer;
  return 0;
}

LoggingReporter* LoggingReporter::DefaultLoggingReporter() {
  static std::unique_ptr<LoggingReporter> error_reporter = std::make_unique<LoggingReporter>();
  return error_reporter.get();
}
}