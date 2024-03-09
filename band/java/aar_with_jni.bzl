load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

def aar_with_jni(
    name,
    android_library,
    headers = None,
    flatten_headers = False):
    """Generates an Android AAR with repo root license given an Android library target.

    Args:
      name: Name of the generated .aar file.
      android_library: The `android_library` target to package. Note that the
          AAR will contain *only that library's .jar` sources. It does not
          package the transitive closure of all Java source dependencies.
      headers: Optional list of headers that will be included in the
          generated .aar file. This is useful for distributing self-contained
          .aars with native libs that can be used directly by native clients.
      flatten_headers: Whether to flatten the output paths of included headers.
    """

    # Generate dummy AndroidManifest.xml for dummy apk usage
    # (dummy apk is generated by <name>_dummy_app_for_so target below)
    native.genrule(
        name = name + "_binary_manifest_generator",
        outs = [name + "_generated_AndroidManifest.xml"],
        cmd = """
cat > $(OUTS) <<EOF
<manifest
  xmlns:android="http://schemas.android.com/apk/res/android"
  package="dummy.package.for.so">
  <uses-sdk android:minSdkVersion="999"/>
</manifest>
EOF
""" # `android:minSdkVersion="999"` means this manifest cannot be used.
    )

    # TODO(widiba03304): Find more efficient way to do this.
    # Generate dummy apk including .so files and later we extract out
    # .so files and throw away the apk.
    android_binary(
        name = name + "_dummy_app_for_so",
        manifest = name + "_generated_AndroidManifest.xml",
        custom_package = "dummy.package.for.so",
        deps = [android_library],
        multidex = "native",
        # In some platforms we don't have an Android SDK/NDK and this target
        # can't be built. We need to prevent the build system from trying to
        # use the target in that case.
        tags = ["manual"],
    )

    srcs = [
        android_library + ".aar",
        name + "_dummy_app_for_so_unsigned.apk",
        "//:LICENSE",
    ]

    cmd = """
cp $(location {0}.aar) $(location :{1}.aar)
chmod +w $(location :{1}.aar)
origdir=$$PWD
cd $$(mktemp -d)
unzip $$origdir/$(location :{1}_dummy_app_for_so_unsigned.apk) "lib/*"
cp -r lib jni
zip -r $$origdir/$(location :{1}.aar) jni/*/*.so
cp $$origdir/$(location //:LICENSE) ./
zip $$origdir/$(location :{1}.aar) LICENSE
""".format(android_library, name)

    if headers:
        srcs += headers
        cmd += """
        mkdir headers
        """
        for src in headers:
            if flatten_headers:
                cmd += """
                    cp -RL $$origdir/$(location {0}) headers/$$(basename $(location {0}))
                """.format(src)
            else:
                cmd += """
                    mkdir -p headers/$$(dirname $(location {0}))
                    cp -RL $$origdir/$(location {0}) headers/$(location {0})
                """.format(src)
        cmd += "zip -r $$origdir/$(location :{0}.aar) headers".format(name)

    native.genrule(
        name = name,
        srcs = srcs,
        outs = [name + ".aar"],
        # In some platforms we don't have an Android SDK/NDK and this target
        # can't be built. We need to prevent the build system from trying to
        # use the target in that case.
        tags = ["manual"],
        cmd = cmd,
    )

def aar_without_jni(
        name,
        android_library):
    """Generates an Android AAR with repo root license given a pure Java Android library target.

    Args:
      name: Name of the generated .aar file.
      android_library: The `android_library` target to package. Note that the
          AAR will contain *only that library's .jar` sources. It does not
          package the transitive closure of all Java source dependencies.
    """

    srcs = [
        android_library + ".aar",
        "//:LICENSE",
    ]

    cmd = """
cp $(location {0}.aar) $(location :{1}.aar)
chmod +w $(location :{1}.aar)
origdir=$$PWD
cd $$(mktemp -d)
cp $$origdir/$(location //:LICENSE) ./
zip $$origdir/$(location :{1}.aar) LICENSE
""".format(android_library, name)

    native.genrule(
        name = name,
        srcs = srcs,
        outs = [name + ".aar"],
        cmd = cmd,
    )
