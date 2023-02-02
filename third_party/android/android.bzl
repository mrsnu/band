"""Android configuration utility"""

_ANDROID_NDK_HOME = "ANDROID_NDK_HOME"
_ANDROID_SDK_HOME = "ANDROID_SDK_HOME"
_ANDROID_NDK_API_LEVEL = "ANDROID_NDK_API_LEVEL"
_ANDROID_SDK_API_LEVEL = "ANDROID_SDK_API_LEVEL"
_ANDROID_BUILD_TOOLS_VERSION = "ANDROID_BUILD_TOOLS_VERSION"

def _init_android_impl(ctx):
    native.android_sdk_repository(
        name = "androidsdk",
        path = ctx.environ.get(_ANDROID_SDK_HOME),
        api_level = ctx.environ.get(_ANDROID_SDK_API_LEVEL),
        build_tools_version = ctx.environ.get(_ANDROID_BUILD_TOOLS_VERSION),
    )

    native.android_ndk_repository(
        name = "androidndk",
        path = ctx.environ.get(_ANDROID_NDK_HOME),
        api_level = ctx.environ.get(_ANDROID_NDK_API_LEVEL),
    )

init_android = repository_rule(
    implementation = _init_android_impl,
    environ = [
        _ANDROID_NDK_HOME,
        _ANDROID_SDK_HOME,
        _ANDROID_NDK_API_LEVEL,
        _ANDROID_SDK_API_LEVEL,
        _ANDROID_BUILD_TOOLS_VERSION,
    ],
)