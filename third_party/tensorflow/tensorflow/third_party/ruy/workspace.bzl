"""Loads the ruy library, used by TensorFlow Lite."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "ruy",
        sha256 = "c68f213a2026e820e08682210da27ee8f138d7c0c98fe589e0378eb21170cc48",
        strip_prefix = "ruy-2.9.2",
        url = "https://github.com/mrsnu/ruy/archive/refs/tags/v2.9.2.zip",
    )
