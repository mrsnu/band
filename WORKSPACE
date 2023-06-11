workspace(name = "org_band")

load("//band:workspace_repo.bzl", "workspace_repo")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_foreign_cc",
    sha256 = "2a4d07cd64b0719b39a7c12218a3e507672b82a97b98c6a89d38565894cf7c51",
    strip_prefix = "rules_foreign_cc-0.9.0",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.9.0.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

_ALL_CONTENT = """
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
"""

http_archive(
    name = "com_nvidia_tensorrt",
    build_file_content = _ALL_CONTENT,
    strip_prefix = "TensorRT-8.6.1",
    url = "https://github.com/NVIDIA/TensorRT/archive/refs/tags/v8.6.1.tar.gz",
)

workspace_repo()

# Note: This repo should lie here to respect TensorFlow's build dependency.
http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-2.9.2_thread_affinity",
    sha256 = "26ec28d160a2c850c019c330e9b8ec280c9c2412d21440c9bde73eef8e0ff1fd",
    url = "https://github.com/mrsnu/tensorflow/archive/refs/tags/v2.9.2_thread_affinity.zip"
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()