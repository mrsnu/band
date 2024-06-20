workspace(name = "org_band")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//band:workspace_repo.bzl", "workspace_repo")

workspace_repo()

# Note: This repo should lie here to respect TensorFlow's build dependency.
http_archive(
    name = "org_tensorflow",
    sha256 = "26ec28d160a2c850c019c330e9b8ec280c9c2412d21440c9bde73eef8e0ff1fd",
    strip_prefix = "tensorflow-2.9.2_thread_affinity",
    url = "https://github.com/mrsnu/tensorflow/archive/refs/tags/v2.9.2_thread_affinity.zip",
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()
