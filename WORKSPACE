workspace(name = "org_band")

load("//third_party/absl:workspace.bzl", absl = "repo")

def _remote_repo_impl(ctx):
    ctx.download_and_extract(
        url = ctx.attr.urls,
        sha256 = ctx.attr.sha256,
        type = ctx.attr.type,
        stripPrefix = ctx.attr.strip_predix,
    )
    patch_files = ctx.attr.patch_files
    if patch_files:
        for patch_file in patch_files:
            patch_file = ctx.path(Label(patch_file)) if patch_file else None
            if patch_file:
                ctx.patch(patch_file, strip = True)

_remote_repo = repository_rule(
    implementation = _remote_repo_impl,
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "patch_files": attr.string_list(),
    }
)

def remote_repository_rule(name, sha256, urls **kwargs):
    _remote_repo(
        name = name,
        sha256 = sha256,
        urls = urls,
        **kwargs,
    )
    
def remote_patch_repository_rule(name, sha256, urls, patch_files, **kwargs):
    _remote_repo(
        name = name,
        sha256 = sha256,
        urls = urls,
        patch_files = patch_files,
    )
