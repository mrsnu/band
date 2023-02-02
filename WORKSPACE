workspace(name = "org_band")

load("//band:workspace.bzl", "band_workspace")
band_workspace()

# TODO(widiba03304): Upload org_tensorflow and change this to http_archive.
local_repository(
    name = "org_tensorflow",
    path = "third_party/tensorflow",
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace2.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace1.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace0.bzl", "workspace")
workspace()

load("//third_party/android:android.bzl", "init_android")
init_android()
