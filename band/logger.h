/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_LOGGER_H_
#define BAND_LOGGER_H_

#include <cstdarg>
#include <functional>
#include <map>
#include <string>

#include "absl/status/status.h"
#include "band/common.h"

namespace band {
/*
  Logger is a singleton class that provides basic thread-safe logging
  functionality. It is used by the BAND_LOG_* macros defined below from internal
  sources.

  The logger can be configured to log at a certain verbosity level, e.g., only
  warnings and errors if its verbosity is set to kWarning. The def
  provides two additional ways to handle log messages. First, the logger can be
  configured to report log messages to user-defined reporter function. Second,
  the logger provides a way to retrieve the last log message via
  GetLastLog().
*/

class Logger {
 public:
  static Logger& Get();

  void SetVerbosity(LogSeverity severity);
  CallbackId SetReporter(
      std::function<void(LogSeverity, const char*)> reporter);
  absl::Status RemoveReporter(CallbackId callback_id);
  std::pair<LogSeverity, std::string> GetLastLog() const;

  // DebugLog is only enabled in debug mode.
  void DebugLog(const char* format, ...);
  void Log(LogSeverity severity, const char* format, ...);

 private:
  void LogFormatted(LogSeverity severity, const char* format, va_list args);

  Logger() = default;
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  CallbackId next_callback_id_ = 0;
  std::map<CallbackId, std::function<void(LogSeverity, const char*)>>
      reporters_;

  LogSeverity verbosity_ = LogSeverity::kInfo;
  std::pair<LogSeverity, std::string> last_message_;
};
}  // namespace band

#ifdef NDEBUG
#define BAND_LOG_DEBUG(format, ...) \
  do {                              \
  } while (false);
#else
#define BAND_LOG_DEBUG(format, ...) \
  band::Logger::Get().DebugLog(format, ##__VA_ARGS__);
#endif

#define BAND_LOG(severity, format, ...) \
  band::Logger::Get().Log(severity, format, ##__VA_ARGS__);

// Convenience macro for logging a statement *once* for a given process lifetime
#define BAND_LOG_ONCE(severity, format, ...)    \
  do {                                          \
    static const bool s_logged = [&] {          \
      BAND_LOG(severity, format, ##__VA_ARGS__) \
      return true;                              \
    }();                                        \
    (void)s_logged;                             \
  } while (false);

#if !defined(__PRETTY_FUNCTION__) && defined(_MSC_VER)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#define BAND_NOT_IMPLEMENTED                                 \
  BAND_LOG(band::LogSeverity::kError, "Not implemented: %s", \
           __PRETTY_FUNCTION__);

#endif