
#include "band/error_reporter.h"

#include <cstdarg>

#include "band/logger.h"

namespace band {

int ErrorReporter::Report(const char* format, ...) const {
  va_list args;
  va_start(args, format);
  int code = Report(format, args);
  va_end(args);
  return code;
}

// TODO(aselle): Make the name of ReportError on engine the same, so
// we can use the ensure functions w/o a engine and w/ a reporter.
int ErrorReporter::ReportError(void*, const char* format, ...) const {
  va_list args;
  va_start(args, format);
  int code = Report(format, args);
  va_end(args);
  return code;
}

int StderrReporter::Report(const char* format, va_list args) const {
  Logger::LogFormatted(LogSeverity::BAND_LOG_ERROR, format, args);
  return 0;
}

ErrorReporter* DefaultErrorReporter() {
  static StderrReporter* error_reporter = new StderrReporter;
  return error_reporter;
}

}  // namespace band