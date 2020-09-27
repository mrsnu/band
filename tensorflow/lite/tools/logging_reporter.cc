#include "tensorflow/lite/tools/logging_reporter.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
int LoggingReporter::Report(const char* format, va_list args) {
  int count = snprintf(nullptr, 0, format, args) + 1;
  char* buffer = new char[count];
  snprintf(buffer, count, format, args);
  TFLITE_LOG(ERROR) << buffer;
  delete[] buffer;
}

LoggingReporter* LoggingReporter::DefaultLoggingReporter() {
  static LoggingReporter* error_reporter = new LoggingReporter;
  return error_reporter;
}
}