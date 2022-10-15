def clean_dep(dep):
    return str(Label(dep))

def band_copts():
    copts = select({
        clean_dep("//band:windows"): [
            "/DBAND_COMPILE_LIBRARY",
            "/wd4018",
        ],
        "//conditions:default": [
            "-Wno-sign-compare",
        ],
    }) + select({
        clean_dep("//band:android"): [
            "-ffunction-sections",
            "-fdata-sections",
        ],
        "//conditions:default": [],
    }) + select({
        clean_dep("//band:windows"): [],
        "//conditions:default": [
            "-fno-exceptions",
        ],
    })
    return copts

def _make_search_paths(prefix, levels_to_root):
    return ",".join(
        [
            "-rpath,%s/%s" % (prefix, "/".join([".."] * search_level))
            for search_level in range(levels_to_root + 1)
        ],
    )

def _rpath_linkopts(name):
    # Search parent directories up to the TensorFlow root directory for shared
    # object dependencies, even if this op shared object is deeply nested
    # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
    # the root and tensorflow/libtensorflow_framework.so should exist when
    # deployed. Other shared object dependencies (e.g. shared between contrib/
    # ops) are picked up as long as they are in either the same or a parent
    # directory in the tensorflow/ tree.
    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        clean_dep("//band:windows"): [],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),
        ],
    })

# Shared libraries have different name pattern on different platforms,
# but cc_binary cannot output correct artifact name yet,
# so we generate multiple cc_binary targets with all name patterns when necessary.
# TODO(pcloudy): Remove this workaround when https://github.com/bazelbuild/bazel/issues/4570
# is done and cc_shared_library is available.
SHARED_LIBRARY_NAME_PATTERNS = [
    "lib%s.so%s",  # On Linux, shared libraries are usually named as libfoo.so
    "%s%s.dll",  # On Windows, shared libraries are usually named as foo.dll
]

def band_linkopts_unstripped():
    """Define linker flags to reduce size of Band binary.
    
        These are useful when trying to investigate the relative size of the
        symbols in Band.

    Returns:
        a select object with proper linkopts
    """

    # In case you wonder why there's no --icf is because the gains were
    # negligible, and created potential compatibility problems.
    return select({
        clean_dep("//band:android"): [
            "-latomic",  # Required for some uses of ISO C++11 <atomic> in x86.
            "-Wl,--no-export-dynamic",  # Only inc syms referenced by dynamic obj.
            "-Wl,--gc-sections",  # Eliminate unused code and data.
            "-Wl,--as-needed",  # Don't link unused libs.
        ],
        "//conditions:default": [],
    })

def band_jni_linkopts_unstripped():
    """Defines linker flags to reduce size of Band binary with JNI.

       These are useful when trying to investigate the relative size of the
       symbols in Band.

    Returns:
       a select object with proper linkopts
    """

    # In case you wonder why there's no --icf is because the gains were
    # negligible, and created potential compatibility problems.
    return select({
        clean_dep("//band:android"): [
            "-latomic",  # Required for some uses of ISO C++11 <atomic> in x86.
            "-Wl,--gc-sections",  # Eliminate unused code and data.
            "-Wl,--as-needed",  # Don't link unused libs.
        ],
        "//conditions:default": [],
    })

def band_symbol_opts():
    """Defines linker flags whether to include symbols or not."""
    return select({
        clean_dep("//band:debug"): [],
        clean_dep("//band:windows"): [],
        "//conditions:default": [
            "-s",  # Omit symbol table, for all non debug builds
        ],
    })

def band_linkopts():
    """Defines linker flags for linking Band binary."""
    return band_linkopts_unstripped() + band_symbol_opts()

def band_jni_linkopts():
    """Defines linker flags for linking Band binary with JNI."""
    return band_jni_linkopts_unstripped() + band_symbol_opts()

def band_cc_android_test(
        name,
        # copts = ["-Wall"] + band_copts(),
        linkopts = band_linkopts() + select({
            clean_dep("//band:android"): [
                "-pie",
                "-lm",
                "-static",
                "-Wl,--rpath=/data/local/tmp/",
            ],
            "//conditions:default": [],
        }),
        linkstatic = select({
            clean_dep("//band:android"): True,
            # linkstatic in cc_test is true for window by default.
            clean_dep("//band:windows"): True,
            "//conditions:default": False,
        }),
        **kwargs):
    """Builds a standalone test for android device when android config is on."""
    native.cc_test(
        name = name,
        # copts = copts,
        linkopts = linkopts,
        linkstatic = linkstatic,
        **kwargs
    )

def band_cc_shared_object(
          name,
          srcs = [],
          deps = [],
          data = [],
          copts = band_copts(),
          linkopts = band_jni_linkopts(),
          soversion = None,
          kernels = [],
          linkstatic = 1,
          per_os_targets = False,
          visibility = None,
          **kwargs):
    """Builds a shared object for Band"""
    if soversion != None:
        suffix = "." + str(soversion).split(".")[0]
        longsuffix = "." + str(soversion)
    else:
        suffix = ""
        longsuffix = ""

    if per_os_targets:
        names = [
            (
                pattern % (name, ""),
                pattern % (name, suffix),
                pattern % (name, longsuffix),
            )
            for pattern in SHARED_LIBRARY_NAME_PATTERNS
        ]
    else:
        names = [(
            name,
            name + suffix,
            name + longsuffix
        )]

    for name_os, name_os_major, name_os_full in names:
        if name_os.endswith(".dll"):
            name_os_major = name_os
            name_os_full = name_os

        if name_os != name_os_major:
            native.genrule(
                name = name_os + "_sym",
                outs = [name_os],
                srcs = [name_os_major],
                output_to_bindir = 1,
                cmd = "ln -sf $$(basename %<) $@",
            )

            native.genrule(
                name = name_os_major + "_sym",
                outs = [name_os_major],
                srcs = [name_os_full],
                output_to_bindir = 1,
                cmd = "ln -sf $$(basename $<) $@",
            )

        soname = name_os_major.split("/")[-1]

        data_extra = []

        native.cc_binary(
            name = name_os_full,
            srcs = srcs,
            deps = deps,
            linkshared = 1,
            data = data + data_extra,
            linkopts = linkopts + _rpath_linkopts(name_os_full) + select({
                clean_dep("//band:windows"): [],
                clean_dep("//band:android"): [
                    "-Wl,-soname," + soname,
                ],
                "//conditions:default": [
                    "-lpthread",
                    "-Wl,-soname," + soname,
                ],
            }),
            visibility = visibility,
            **kwargs
        )

    flat_names = [item for sublist in names for item in sublist]
    if name not in flat_names:
        native.filegroup(
            name = name,
            srcs = select({
                clean_dep("//band:windows"): [":%s.dll" % (name)],
                "//conditions:default": [":lib%s.so%s" % (name, longsuffix)],
            }),
            visibility = visibility,
        )