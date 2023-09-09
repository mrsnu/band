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

#include "absl/strings/str_format.h"


#ifdef __ANDROID__
#include <android/log.h>

int LogSeverityToAndroid(band::LogSeverity severity) {
  switch (severity) {
    case band::LogSeverity::kInfo:
      return ANDROID_LOG_INFO;
      break;
    case band::LogSeverity::kWarning:
      return ANDROID_LOG_WARN;
      break;
    case band::LogSeverity::kError:
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

Logger& Logger::Get() {
  static Logger* logger = new Logger;
  return *logger;
}

CallbackId Logger::SetReporter(
    std::function<void(LogSeverity, const char*)> reporter) {
  CallbackId id = next_callback_id_++;
  reporters_[id] = reporter;
  return id;
}

absl::Status Logger::RemoveReporter(CallbackId callback_id) {
  if (reporters_.find(callback_id) == reporters_.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The given callback id does not exist. %d",
                        static_cast<int>(callback_id)));
  }
  reporters_.erase(callback_id);
  return absl::OkStatus();
}

std::pair<LogSeverity, std::string> Logger::GetLastMessage() const {
  return last_message_;
}

void Logger::DebugLog(const char* format, ...) {
  static std::mutex log_mutex;
  std::lock_guard<std::mutex> lock(log_mutex);
  va_list args;
  va_start(args, format);
  LogFormatted(LogSeverity::kInfo, format, args);
  va_end(args);
}

void Logger::Log(LogSeverity severity, const char* format, ...) {
  if (verbosity_ <= severity) {
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    va_list args;
    va_start(args, format);
    LogFormatted(severity, format, args);
    va_end(args);

    for (const auto& reporter : reporters_) {
      reporter.second(severity, format);
    }

    last_message_ = std::make_pair(severity, format);
  }
}

void Logger::LogFormatted(LogSeverity severity, const char* format,
                          va_list args) {
  static std::mutex log_mutex;
  std::lock_guard<std::mutex> lock(log_mutex);
  fprintf(stderr, "%s: ", ToString(severity));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
  vfprintf(stderr, format, args);
#pragma clang diagnostic pop
  fputc('\n', stderr);

#ifdef __ANDROID__
  __android_log_vprint(LogSeverityToAndroid(severity), "BAND", format, args);
#endif
}

void Logger::SetVerbosity(LogSeverity severity) { verbosity_ = severity; }

}  // namespace band