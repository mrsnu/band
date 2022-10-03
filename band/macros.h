#ifndef BAND_MACROS_H_
#define BAND_MACROS_H_

#ifdef __has_builtin
#define BAND_HAS_BUILTIN(x) __has_builtin(x)
#else
#define BAND_HAS_BUILTIN(x) 0
#endif

// Compilers can be told that a certain branch is not likely to be taken
// (for instance, a CHECK failure), and use that information in static
// analysis. Giving it this information can help it optimize for the
// common case in the absence of better information (ie.
// -fprofile-arcs).
#if BAND_HAS_BUILTIN(__builtin_expect) || (defined(__GNUC__) && __GNUC__ >= 3)
#define BAND_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define BAND_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define BAND_PREDICT_FALSE(x) (x)
#define BAND_PREDICT_TRUE(x) (x)
#endif

#endif  // BAND_MACROS_H_