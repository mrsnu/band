
#include "band/logger.h"

#include <cstdarg>
#include <cstdio>

#ifdef __ANDROID__
#include <android/log.h>

int LogSeverityToAndroid(band::LogSeverity severity) {
  switch (severity) {
    case band::BAND_LOG_INFO:
      return ANDROID_LOG_INFO;
      break;
    case band::BAND_LOG_WARNING:
      return ANDROID_LOG_WARN;
      break;
    case band::BAND_LOG_ERROR:
      return ANDROID_LOG_ERROR;
      break;
  }
  return -1;
}
#endif

namespace band {
LogSeverity Logger::verbosity = BAND_LOG_INFO;

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
  fprintf(stderr, "%s: ", GetSeverityName(severity));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
  vfprintf(stderr, format, args);
#pragma clang diagnostic pop
  fputc('\n', stderr);

#ifdef __ANDROID__
  __android_log_vprint(LogSeverityToAndroid(severity), "BAND", format, args);
#endif
}

void Logger::SetVerbosity(int severity) {
  verbosity = static_cast<LogSeverity>(severity);
}

const char* Logger::GetSeverityName(LogSeverity severity) {
  switch (severity) {
    case BAND_LOG_INFO:
      return "INFO";
    case BAND_LOG_WARNING:
      return "WARNING";
    case BAND_LOG_ERROR:
      return "ERROR";
  }
  return "<Unknown severity>";
}

}  // namespace band