
#ifndef BAND_ERROR_REPORTER_H_
#define BAND_ERROR_REPORTER_H_

#include <cstdarg>

namespace band {

/// A functor that reports error to supporting system. Invoked similar to
/// printf.
///
/// Usage:
///  ErrorReporter foo;
///  foo.Report("test %d", 5);
/// or
///  va_list args;
///  foo.Report("test %d", args); // where args is va_list
///
/// Subclass ErrorReporter to provide another reporting destination.
/// For example, if you have a GUI program, you might redirect to a buffer
/// that drives a GUI error log box.
class ErrorReporter {
 public:
  virtual ~ErrorReporter() {}
  virtual int Report(const char* format, va_list args) const = 0;
  int Report(const char* format, ...) const;
  int ReportError(void*, const char* format, ...) const;
};

// An error reporter that simply writes the message to stderr.
struct StderrReporter : public ErrorReporter {
  int Report(const char* format, va_list args) const override;
};

// Return the default error reporter (output to stderr).
ErrorReporter* DefaultErrorReporter();

}  // namespace band

#if !defined(__PRETTY_FUNCTION__) && defined(_WIN32)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

// You should not make bare calls to the error reporter, instead use the
// BAND_REPORT_ERROR macro, since this allows message strings to be
// stripped when the binary size has to be optimized. If you are looking to
// reduce binary size, define BAND_STRIP_ERROR_STRINGS when compiling and
// every call will be stubbed out, taking no memory.
#ifndef BAND_STRIP_ERROR_STRINGS
#define BAND_REPORT_ERROR(reporter, ...) \
  do {                                   \
    reporter->Report(__VA_ARGS__);       \
  } while (false)
#else  // BAND_STRIP_ERROR_STRINGS
#define BAND_REPORT_ERROR(reporter, ...)
#endif  // BAND_STRIP_ERROR_STRINGS

#define BAND_NOT_IMPLEMENTED                                        \
  BAND_REPORT_ERROR(                                                \
      DefaultErrorReporter(),                                       \
      "%s at \n line number %d in file %s is not implemented yet.", \
      __PRETTY_FUNCTION__, __LINE__, __FILE__)

#endif  // BAND_ERROR_REPORTER_H_
