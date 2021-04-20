"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    native.local_repository(
        name = "ruy",
        path = "third_party/ruy/ruy",
    )