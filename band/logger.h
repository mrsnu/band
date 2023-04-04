#ifndef BAND_LOGGER_H_
#define BAND_LOGGER_H_

#include <cstddef>
#include <cstdarg>

namespace band {

enum class LogSeverity : size_t {
  DEBUG = 0,
  INTERNAL = 1,
  INFO = 2,
  WARNING = 3,
  ERROR = 4,
};

class Logger {
 public:
  static void SetVerbosity(LogSeverity severity);

  static void Log(LogSeverity severity, const char* format, ...);
  static void LogFormatted(LogSeverity severity, const char* format,
                           va_list args);

 private:
  // Only accept logs with higher severity than verbosity level.
  static LogSeverity verbosity;
  static const char* GetSeverityName(LogSeverity severity);
};

}  // namespace band

#define BAND_LOG_IMPL(severity, format, ...) \
  band::Logger::Log(severity, format, ##__VA_ARGS__);

#ifndef NDEBUG
#define BAND_LOG_DEBUG(format, ...) \
  BAND_LOG_IMPL(band::LogSeverity::DEBUG, format, ##__VA_ARGS__);
#else
#define BAND_LOG_DEBUG(format, ...) ;
#endif  // NDEBUG

#define BAND_LOG_INTERNAL(format, ...) \
  BAND_LOG_IMPL(band::LogSeverity::INTERNAL, format, ##__VA_ARGS__);

#define BAND_LOG_INFO(format, ...) \
  BAND_LOG_IMPL(band::LogSeverity::INFO, format, ##__VA_ARGS__);

#define BAND_LOG_WARNING(format, ...) \
  BAND_LOG_IMPL(band::LogSeverity::WARNING, format, ##__VA_ARGS__);

#define BAND_LOG_ERROR(format, ...) \
  BAND_LOG_IMPL(band::LogSeverity::ERROR, format, ##__VA_ARGS__);

#define BAND_LOG_ONCE(severity, format, ...)    \
  do {                                               \
    static const bool s_logged = [&] {               \
      BAND_LOG_IMPL(severity, format, ##__VA_ARGS__) \
      return true;                                   \
    }();                                             \
    (void)s_logged;                                  \
  } while (false);

#endif  // BAND_LOGGER_H_