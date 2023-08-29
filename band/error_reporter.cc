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