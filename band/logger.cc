// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "band/logger.h"

#include <cstdarg>
#include <cstdio>
#include <mutex>

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
    case band::BAND_LOG_NUM_SEVERITIES:
    default:
      break;
  }
  return -1;
}
#endif

namespace band {
LogSeverity Logger::verbosity = BAND_LOG_INFO;

void Logger::Log(LogSeverity severity, const char* format, ...) {
  if (verbosity <= severity) {
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    va_list args;
    va_start(args, format);
    LogFormatted(severity, format, args);
    va_end(args);
  }
}

void Logger::LogFormatted(LogSeverity severity, const char* format,
                          va_list args) {
  static std::mutex log_mutex;
  std::lock_guard<std::mutex> lock(log_mutex);
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
    case BAND_LOG_NUM_SEVERITIES:
    default:
      return "<Unknown severity>";
  }
}

}  // namespace band