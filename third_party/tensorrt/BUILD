load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

cmake(
    name = "tensorrt",
    lib_source = "@com_nvidia_tensorrt//:all_srcs",
    out_static_libs = ["libtensorrt.a"],
)