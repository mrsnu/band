# reference: tensorflow/tensorflow.bzl
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

def clean_dep(dep):
    return str(Label(dep))

def if_tfl(a):
    return select({
        clean_dep("//band:tflite"): a,
        "//conditions:default": [],
    })

def band_copts():
    return (
        if_tfl(["-DBAND_BACKEND_TFL"])
    )

def band_cc_android_test(
        name,
        copts = ["-Wall"] + tflite_copts(),
        linkopts = tflite_linkopts() + select({
            clean_dep("//tensorflow:android"): [
                "-pie",
                "-lm",
                "-static",
                "-Wl,--rpath=/data/local/tmp/",
            ],
            "//conditions:default": [],
        }),
        linkstatic = select({
            clean_dep("//tensorflow:android"): 1,
            "//conditions:default": 0,
        }),
        **kwargs):
    """Builds a standalone test for android device when android config is on."""
    native.cc_test(
        name = name,
        copts = copts,
        linkopts = linkopts,
        linkstatic = linkstatic,
        **kwargs
    )
