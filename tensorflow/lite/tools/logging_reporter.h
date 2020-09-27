#ifndef TENSORFLOW_LITE_TOOLS_LOGGING_REPORTER_H_
#define TENSORFLOW_LITE_TOOLS_LOGGING_REPORTER_H_

#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {
// An error reporter that simply writes the message to logging.
struct LoggingReporter : public ErrorReporter {
  int Report(const char* format, va_list args) override;
  static LoggingReporter* DefaultLoggingReporter();
};
} // namespace tflite

#endif // TENSORFLOW_LITE_TOOLS_LOGGING_REPORTER_H_