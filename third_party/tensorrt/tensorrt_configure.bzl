"""TensorRT configuration utility"""

_TENSORRT_INCLUDE_PATH = "TENSORRT_INCLUDE_PATH"
_TENSORRT_LIB_PATH = "TENSORRT_LIB_PATH"

_COPY_RULES = "COPY_RULES"

_TENSORRT_LIBS = [
    "libnvinfer.so",
    "libnvinfer_plugin.so",
]

_TENSORRT_HEADERS = [
    "NvInfer.h",
    "NvUtils.h",
    "NvInferPlugin.h",
]

def copy_files_rule(repository_ctx, name, srcs, outs):
    """Returns a rule to copy a set of files."""
    cmds = []

    # Copy files.
    for src, out in zip(srcs, outs):
        cmds.append('cp -f "%s" "$(location %s)"' % (src, out))
    outs = [('        "%s",' % out) for out in outs]
    return """genrule(
    name = "%s",
    outs = [
%s
    ],
    cmd = \"""%s \""",
)""" % (name, "\n".join(outs), " && \\\n".join(cmds))

def _tensorrt_configure_impl(ctx):
    copy_rules = []
    copy_rules.append(
        copy_files_rule(
            ctx,
            name = "tensorrt_include",
            srcs = [ctx.os.environ.get(_TENSORRT_INCLUDE_PATH) + "/" + header for header in _TENSORRT_HEADERS],
            outs = ["tensorrt/include/" + header for header in _TENSORRT_HEADERS]
        )
    )
    copy_rules.append(
        copy_files_rule(
            ctx,
            name = "tensorrt_lib",
            srcs = [ctx.os.environ.get(_TENSORRT_LIB_PATH) + "/" + lib for lib in _TENSORRT_LIBS],
            outs = ["tensorrt/lib/" + lib for lib in _TENSORRT_LIBS]
        )
    )

    ctx.template(
        "BUILD",
        Label("//third_party/tensorrt:BUILD.tpl"),
        {
            _COPY_RULES: "\n".join(copy_rules),
        }
    )

tensorrt_configure = repository_rule(
    implementation = _tensorrt_configure_impl,
    environ = [
        "TENSORRT_INCLUDE_PATH",
        "TENSORRT_LIB_PATH",
    ]
)