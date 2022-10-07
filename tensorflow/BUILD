# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

load("@bazel_skylib//lib:selects.bzl", "selects")
load(
    "//tensorflow:tensorflow.bzl",
    "VERSION",
    "if_google",
    "if_oss",
    "tf_cc_shared_object",
    "tf_custom_op_library_additional_deps_impl",
    "tf_native_cc_binary",
)
package(
    default_visibility = [":internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "LICENSE",
    "ACKNOWLEDGMENTS",
    # The leakr files are used by //third_party/cloud_tpu and
    # //third_party/tensorboard/google:copybara_config_test.
    "leakr_badwords.dic",
    "leakr_badfiles.dic",
    "leakr_file_type_recipe.ftrcp",
    "libtensorflow_framework.so.2",
])

# Config setting used when building for products
# which requires restricted licenses to be avoided.
config_setting(
    name = "no_lgpl_deps",
    define_values = {"__TENSORFLOW_NO_LGPL_DEPS__": "1"},
    visibility = ["//visibility:public"],
)

# Config setting that disables the default logger, only logging
# to registered TFLogSinks
config_setting(
    name = "no_default_logger",
    define_values = {"no_default_logger": "true"},
    visibility = ["//visibility:public"],
)

# Config setting for determining if we are building for Android.
config_setting(
    name = "android",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "android"},
        {},
    ),
    values = if_oss(
        {"crosstool_top": "//external:android/crosstool"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "android"},
        {},
    ),
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
        cpu = "x86",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86_64",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "android"},
        {},
    ),
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
        cpu = "x86_64",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_armeabi",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "android"},
        {},
    ),
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
        cpu = "armeabi",
    ),
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# config_setting(
#     name = "chromiumos_x86_64",
#     flag_values = {"//tools/cpp:cc_target_os": "chromiumos"},
#     values = {"cpu": "k8"},
#     visibility = ["//visibility:public"],
# )
#
# config_setting(
#     name = "chromiumos_arm64",
#     flag_values = {"//tools/cpp:cc_target_os": "chromiumos"},
#     values = {"cpu": "arm"},
#     visibility = ["//visibility:public"],
# )
#
# config_setting(
#     name = "chromiumos_armv7",
#     flag_values = {"//tools/cpp:cc_target_os": "chromiumos"},
#     values = {"cpu": "armeabi-v7a"},
#     visibility = ["//visibility:public"],
# )
# copybara:uncomment_end

config_setting(
    name = "emscripten",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "emscripten"},
        {},
    ),
    values = if_oss(
        {"crosstool_top": "//external:android/emscripten"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "raspberry_pi_armeabi",
    values = {
        "crosstool_top": "@local_config_arm_compiler//:toolchain",
        "cpu": "armeabi",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "android"},
        {},
    ),
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
        cpu = "armeabi-v7a",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "android"},
        {},
    ),
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
        cpu = "arm64-v8a",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_mips",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "mips",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_mips64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "mips64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    # Internal builds query the target OS.
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "windows"},
        {},
    ),
    # OSS builds query the CPU type.
    values = if_oss(
        {"cpu": "x64_windows"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_tensorflow_py_deps",
    define_values = {"no_tensorflow_py_deps": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_x86_64",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "apple"},
        {},
    ),
    values = {
        "apple_platform_type": "macos",
        "cpu": if_google("darwin_x86_64", "darwin"),
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_arm64",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "apple"},
        {},
    ),
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64",
    },
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "macos",
    match_any = [
        ":macos_x86_64",
        ":macos_arm64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "apple"},
        {},
    ),
    values = if_oss(
        {"apple_platform_type": "ios"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "fuchsia",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "fuchsia"},
        {},
    ),
    values = if_oss(
        # TODO(b/149248802) When we have a Fuchsia Bazel SDK update to use the values it sets.
        {"cpu": "fuchsia"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "fuchsia_x86_64",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "fuchsia"},
        {},
    ),
    values = {
        "cpu": "x86_64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios_x86_64",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "apple"},
        {},
    ),
    values = dict(
        if_oss(
            {"crosstool_top": "//tools/osx/crosstool:crosstool"},
        ),
        cpu = "ios_x86_64",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "chromiumos",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "chromiumos"},
        {},
    ),
    values = if_oss(
        {"crosstool_top": "//external:android/chromiumos"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_aarch64",
    values = {"cpu": "aarch64"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_armhf",
    values = {"cpu": "armhf"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_ppc64le",
    values = {"cpu": "ppc"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_s390x",
    values = {"cpu": "s390x"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_mips64",
    values = {"cpu": "mips64"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_riscv64",
    values = {"cpu": "riscv64"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "debug",
    values = {
        "compilation_mode": "dbg",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "optimized",
    values = {
        "compilation_mode": "opt",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "arm",
    values = {"cpu": "arm"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "armeabi",
    values = {"cpu": "armeabi"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "armeabi-v7a",
    values = {"cpu": "armeabi-v7a"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "arm64-v8a",
    values = {"cpu": "arm64-v8a"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "arm_any",
    match_any = [
        ":arm",
        ":armeabi",
        ":armeabi-v7a",
        ":arm64-v8a",
        ":linux_aarch64",
        ":linux_armhf",
    ],
)

config_setting(
    name = "freebsd",
    values = {"cpu": "freebsd"},
    visibility = ["//visibility:public"],
)

# Features that are default ON are handled differently below.
#
config_setting(
    name = "no_aws_support",
    define_values = {"no_aws_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_gcp_support",
    define_values = {"no_gcp_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_hdfs_support",
    define_values = {"no_hdfs_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_nccl_support",
    define_values = {"no_nccl_support": "true"},
    visibility = ["//visibility:public"],
)

# Experimental features
config_setting(
    name = "stackdriver_support",
    define_values = {"stackdriver_support": "true"},
    visibility = ["//visibility:public"],
)

# Crosses between platforms and file system libraries not supported on those
# platforms due to limitations in nested select() statements.
config_setting(
    name = "with_cuda_support_windows_override",
    define_values = {"using_cuda_nvcc": "true"},
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_xla_support",
    define_values = {"with_xla_support": "true"},
    visibility = ["//visibility:public"],
)

# By default, XLA GPU is compiled into tensorflow when building with
# --config=cuda even when `with_xla_support` is false. The config setting
# here allows us to override the behavior if needed.
config_setting(
    name = "no_xla_deps_in_cuda",
    define_values = {"no_xla_deps_in_cuda": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_numa_support",
    define_values = {"with_numa_support": "true"},
    visibility = ["//visibility:public"],
)

# Crosses between framework_shared_object and a bunch of other configurations
# due to limitations in nested select() statements.
config_setting(
    name = "framework_shared_object",
    define_values = {
        "framework_shared_object": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_with_framework_shared_object",
    define_values = {
        "framework_shared_object": "true",
    },
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_cuda_clang",
    define_values = {
        "using_cuda_clang": "true",
    },
)

# Flag to indicate open source build, .bazelrc always has it set to be true
config_setting(
    name = "oss",
    define_values = {
        "open_source_build": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_cuda_clang_with_dynamic_build",
    define_values = {
        "using_cuda_clang": "true",
        "framework_shared_object": "true",
    },
)

config_setting(
    name = "build_oss_using_cuda_clang",
    define_values = {
        "using_cuda_clang": "true",
        "open_source_build": "true",
    },
)

# Setting to use when loading kernels dynamically
config_setting(
    name = "dynamic_loaded_kernels",
    define_values = {
        "dynamic_loaded_kernels": "true",
        "framework_shared_object": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_cuda_nvcc",
    define_values = {
        "using_cuda_nvcc": "true",
    },
)

config_setting(
    name = "using_cuda_nvcc_with_dynamic_build",
    define_values = {
        "using_cuda_nvcc": "true",
        "framework_shared_object": "true",
    },
)

config_setting(
    name = "build_oss_using_cuda_nvcc",
    define_values = {
        "using_cuda_nvcc": "true",
        "open_source_build": "true",
    },
)

config_setting(
    name = "using_rocm_hipcc",
    define_values = {
        "using_rocm_hipcc": "true",
    },
)

config_setting(
    name = "override_eigen_strong_inline",
    values = {"define": "override_eigen_strong_inline=true"},
    visibility = ["//visibility:public"],
)

# This flag is set from the configure step when the user selects with nGraph option.
# By default it should be false
config_setting(
    name = "with_ngraph_support",
    values = {"define": "with_ngraph_support=true"},
    visibility = ["//visibility:public"],
)

# This flag specifies whether TensorFlow 2.0 API should be built instead
# of 1.* API. Note that TensorFlow 2.0 API is currently under development.
config_setting(
    name = "api_version_2",
    define_values = {"tf_api_version": "2"},
    visibility = ["//visibility:public"],
)

# This flag is defined for select statements that match both
# on 'windows' and 'api_version_2'. In this case, bazel requires
# having a flag which is a superset of these two.
config_setting(
    name = "windows_and_api_version_2",
    define_values = {"tf_api_version": "2"},
    values = {"cpu": "x64_windows"},
)

# This flag enables experimental MLIR support.
config_setting(
    name = "with_mlir_support",
    values = {"define": "with_mlir_support=true"},
    visibility = ["//visibility:public"],
)

# This flag enables experimental TPU support
config_setting(
    name = "with_tpu_support",
    values = {"define": "with_tpu_support=true"},
    visibility = ["//visibility:public"],
)

# Specifies via a config setting if this is a mobile build or not, makes
# it easier to combine settings later.
selects.config_setting_group(
    name = "mobile",
    match_any = [
        ":android",
        ":chromiumos",
        ":emscripten",
        ":ios",
    ],
)

config_setting(
    name = "lite_protos_legacy",
    values = {"define": "TENSORFLOW_PROTOS=lite"},
    visibility = ["//visibility:private"],
)

config_setting(
    name = "full_protos",
    values = {"define": "TENSORFLOW_PROTOS=full"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "lite_protos",
    match_any = [":lite_protos_legacy"],
)

selects.config_setting_group(
    name = "mobile_lite_protos",
    match_all = [
        ":lite_protos",
        ":mobile",
    ],
)

selects.config_setting_group(
    name = "mobile_full_protos",
    match_all = [
        ":full_protos",
        ":mobile",
    ],
)

# DO NOT ADD ANY NEW EXCEPTIONS TO THIS LIST!
# Instead, please use public APIs or public build rules TF provides.
# If you need functionality that is not exposed, we will work with you to expand our public APIs.
package_group(
    name = "internal",
    packages = [
        # To pass open source testing in the pip Kokoros.
        "//bazel_pip/tensorflow/...",
        "//learning/brain/swift/x10/...",
        "//perftools/accelerators/xprof/api/...",
        "//third_party/py/autograph/...",
        "//third_party/swift/tensorflow/x10/...",
        "//third_party/swift/tensorflow_apis/...",
        "//tensorflow/...",
        "//tensorflow_estimator/python/estimator/...",
        "//tensorflow_models/official/...",
    ],
)
