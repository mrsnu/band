#ifndef BAND_LOGGER_H_
#define BAND_LOGGER_H_

#include <cstdarg>

namespace Band {
enum LogSeverity {
  BAND_LOG_INFO = 0,
  BAND_LOG_WARNING = 1,
  BAND_LOG_ERROR = 2,
};

class Logger {
public:
  static void SetVerbosity(int severity);

  // Logging hook that takes variadic args.
  static void Log(LogSeverity severity, const char *format, ...);

  // Logging hook that takes a formatted va_list.
  static void LogFormatted(LogSeverity severity, const char *format,
                           va_list args);

private:
  // Only accept logs with higher severity than verbosity level.
  static LogSeverity verbosity;
  static const char *GetSeverityName(LogSeverity severity);
};
} // namespace Band

// Convenience macro for basic internal logging in production builds.
// Note: This should never be used for debug-type logs, as it will *not* be
// stripped in release optimized builds. In general, prefer the error reporting
// APIs for developer-facing errors, and only use this for diagnostic output
// that should always be logged in user builds.
#define BAND_LOG_PROD(severity, format, ...)                                   \
  Band::Logger::Log(severity, format, ##__VA_ARGS__);

// Convenience macro for logging a statement *once* for a given process lifetime
// in production builds.
#define BAND_LOG_PROD_ONCE(severity, format, ...)                              \
  do {                                                                         \
    static const bool s_logged = [&] {                                         \
      BAND_LOG_PROD(severity, format, ##__VA_ARGS__)                           \
      return true;                                                             \
    }();                                                                       \
    (void)s_logged;                                                            \
  } while (false);

#ifndef NDEBUG
// In debug builds, always log.
#define BAND_LOG_INTERNAL BAND_LOG_PROD
#define BAND_LOG_ONCE BAND_LOG_PROD_ONCE
#else
// In prod builds, never log, but ensure the code is well-formed and compiles.
#define BAND_LOG_INTERNAL(severity, format, ...)                               \
  while (false) {                                                              \
    BAND_LOG_PROD(severity, format, ##__VA_ARGS__);                            \
  }
#define BAND_LOG_ONCE(severity, format, ...) ({ ; })
#endif

#endif