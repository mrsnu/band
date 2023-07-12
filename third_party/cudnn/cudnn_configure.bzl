"""TensorRT configuration utility"""

load("//third_party:common.bzl", "copy_files_rule")

_CUDNN_INCLUDE_PATH = "CUDNN_INCLUDE_PATH"
_CUDNN_LIB_PATH = "CUDNN_LIB_PATH"

_COPY_RULES = "COPY_RULES"

_CUDNN_LIBS = [
    "libcudnn.so",
    "libcudnn_adv_infer.so",
    "libcudnn_adv_train.so",
    "libcudnn_cnn_infer.so",
    "libcudnn_cnn_train.so",
    "libcudnn_ops_infer.so",
    "libcudnn_ops_train.so",
]

_CUDNN_HEADERS = [
    "cudnn_backend.h",
    "cudnn_adv_infer.h",
    "cudnn_adv_train.h",
    "cudnn_cnn_infer.h",
    "cudnn_cnn_train.h",
    "cudnn_ops_infer.h",
    "cudnn_ops_train.h",
    "cudnn_version.h",
]


def _cudnn_configure_impl(ctx):
    copy_rules = []
    copy_rules.append(
        copy_files_rule(
            ctx,
            name = "cudnn_include",
            srcs = [ctx.os.environ.get(_CUDNN_INCLUDE_PATH) + "/" + header for header in _CUDNN_HEADERS],
            outs = ["cudnn/include/" + header for header in _CUDNN_HEADERS]
        )
    )
    copy_rules.append(
        copy_files_rule(
            ctx,
            name = "cudnn_lib",
            srcs = [ctx.os.environ.get(_CUDNN_LIB_PATH) + "/" + lib for lib in _CUDNN_LIBS],
            outs = ["cudnn/lib/" + lib for lib in _CUDNN_LIBS]
        )
    )

    ctx.template(
        "BUILD",
        Label("//third_party/cudnn:BUILD.tpl"),
        {
            _COPY_RULES: "\n".join(copy_rules),
        }
    )

cudnn_configure = repository_rule(
    implementation = _cudnn_configure_impl,
    environ = [
        "CUDNN_INCLUDE_PATH",
        "CUDNN_LIB_PATH",
    ]
)