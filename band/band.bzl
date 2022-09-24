# reference: tensorflow/tensorflow.bzl

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
