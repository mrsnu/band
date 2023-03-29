workspace(name = "org_band")

load("//band:workspace_repo.bzl", "workspace_repo")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

workspace_repo()

# Note: This repo should lie here to respect TensorFlow's build dependency.
http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-2.9.2_thread_affinity",
    sha256 = "2b1b1ec8456b692ad7e867499b09b1057cc169897ddf29dfc8be3017d39d596e",
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
