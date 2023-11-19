load("//band:band.bzl", "band_cc_android_test", "band_cc_library")

band_cc_library(
    name = "device_util",
    srcs = [
      "device_util.h",
    ],
)

band_cc_android_test(
    name = "frequency_test",
    size = "small",
    srcs = ["frequency_test.cc"],
    deps = [
        ":device_util",
        "//band/device:frequency",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "thermal_test",
    size = "small",
    srcs = ["thermal_test.cc"],
    deps = [
        ":device_util",
        "//band/device:thermal",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "thermal_profiler_test",
    size = "small",
    srcs = ["thermal_profiler_test.cc"],
    data = [],
    deps = [
        ":device_util",
        "//band/profiler:thermal_profiler",
        "@com_google_googletest//:gtest",
    ],
)