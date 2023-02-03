workspace(name = "org_band")

load("//band:workspace_repo.bzl", "workspace_repo")

workspace_repo()

# Note: This repo should lie here to respect TensorFlow's build dependency.
local_repository(
    name = "org_tensorflow",
    path = "third_party/tensorflow/tensorflow",
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()
