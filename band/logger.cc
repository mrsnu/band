
#include "band/logger.h"

#include <cstdarg>
#include <cstdio>

namespace band {

#ifndef NDEBUG
LogSeverity Logger::verbosity = LogSeverity::INFO;
#else
LogSeverity Logger::verbosity = LogSeverity::DEBUG;
#endif  // NDEBUG

void Logger::Log(LogSeverity severity, const char* format, ...) {
  if (verbosity <= severity) {
    va_list args;
    va_start(args, format);
    LogFormatted(severity, format, args);
    va_end(args);
  }
}

void Logger::LogFormatted(LogSeverity severity, const char* format,
                          va_list args) {
  fprintf(stdout, "[Band][%s]: ", GetSeverityName(severity));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
  vfprintf(stdout, format, args);
#pragma clang diagnostic pop
  fputc('\n', stdout);
}

void Logger::SetVerbosity(LogSeverity severity) {
  verbosity = severity;
}

const char* Logger::GetSeverityName(LogSeverity severity) {
  switch (severity) {
    case LogSeverity::DEBUG:
      return "DEBUG";
    case LogSeverity::INTERNAL:
      return "INTERNAL";
    case LogSeverity::INFO:
      return "INFO";
    case LogSeverity::WARNING:
      return "WARNING";
    case LogSeverity::ERROR:
      return "ERROR";
    default:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

}  // namespace band