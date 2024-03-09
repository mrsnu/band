"""Initialize Band workspace"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/android:android.bzl", android = "repo")

def workspace():
    """Workspace initialization for dependencies."""

    ######## Remote repositories ########
    http_archive(
        name = "com_google_absl",
        url = "https://github.com/abseil/abseil-cpp/archive/refs/tags/20230125.0.tar.gz",
        sha256 = "3ea49a7d97421b88a8c48a0de16c16048e17725c7ec0f1d3ea2683a2a75adc21",
        strip_prefix = "abseil-cpp-20230125.0",
    )

    http_archive(
        name = "build_bazel_rules_android",
        url = "https://github.com/bazelbuild/rules_android/archive/v0.1.1.zip",
        sha256 = "cd06d15dd8bb59926e4d65f9003bfc20f9da4b2519985c27e190cddc8b7a7806",
        strip_prefix = "rules_android-0.1.1",
    )

    http_archive(
        name = "jsoncpp",
        url = "https://github.com/open-source-parsers/jsoncpp/archive/1.9.5.tar.gz",
        sha256 = "f409856e5920c18d0c2fb85276e24ee607d2a09b5e7d5f0a371368903c275da2",
        strip_prefix = "jsoncpp-1.9.5",
    )

    git_repository(
        name = "chrome_tracer",
        commit = "66394d43c894ee26995d3c4fe7f9c33a9e786bdb",
        remote = "https://github.com/mrsnu/chrome-tracer.git",
    )

    http_archive(
        name = "stblib",
        strip_prefix = "stb-b42009b3b9d4ca35bc703f5310eedc74f584be58",
        sha256 = "13a99ad430e930907f5611325ec384168a958bf7610e63e60e2fd8e7b7379610",
        urls = ["https://github.com/nothings/stb/archive/b42009b3b9d4ca35bc703f5310eedc74f584be58.tar.gz"],
        build_file = Label("//third_party/stblib:BUILD"),
    )

    http_archive(
        name = "libyuv",
        urls = ["https://github.com/mrsnu/libyuv/archive/refs/tags/v1.0.0.zip"],
        sha256 = "27ea6ddea93fefcdacb8175c0627c896488c3aebdcd7efd97e4cb2972a316195",
        strip_prefix = "libyuv-1.0.0",
        build_file = Label("//third_party/libyuv:BUILD"),
    )

    http_archive(
        name = "eigen3",
        urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"],
        strip_prefix = "eigen-3.4.0",
        build_file = Label("//third_party/eigen3:eigen_archive.BUILD"),
    )

    ######## Android repositories ########
    android(name = "android_repo")

workspace_repo = workspace
