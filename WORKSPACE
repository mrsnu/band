workspace(name = "org_band")

load("//band:workspace_repo.bzl", "workspace_repo")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

workspace_repo()

# Note: This repo should lie here to respect TensorFlow's build dependency.
# Original
# http_archive(
#     name = "org_tensorflow",
#     strip_prefix = "tensorflow-2.9.2_thread_affinity",
#     sha256 = "26ec28d160a2c850c019c330e9b8ec280c9c2412d21440c9bde73eef8e0ff1fd",
#     url = "https://github.com/mrsnu/tensorflow/archive/refs/tags/v2.9.2_thread_affinity.zip"
# )

# For TF v2.10
http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-band-0.0.1",
    url = "https://github.com/dongho-Han/tensorflow-band/archive/refs/tags/v0.0.1.zip"
)

# local_repository(
#     name = "org_tensorflow",
#     path = "/tf_base/tensorflow",
# )

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()
