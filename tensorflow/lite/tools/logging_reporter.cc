#include "tensorflow/lite/tools/logging_reporter.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
int LoggingReporter::Report(const char* format, va_list args) {
  int count = vsnprintf(nullptr, 0, format, args) + 1;
  char* buffer = new char[count];
  vsnprintf(buffer, count, format, args);
  TFLITE_LOG(ERROR) << buffer;
  delete[] buffer;
  return 0;
}

LoggingReporter* LoggingReporter::DefaultLoggingReporter() {
  static LoggingReporter error_reporter = LoggingReporter();
  return &error_reporter;
}
}