import os
import shutil


# Keep track of symlinks created so on_post_build can clean them up.
_created_symlinks: list[str] = []


def on_pre_build(config, **kwargs):
    """
    Hook that runs before the MkDocs build process starts.

    Creates temporary symbolic links inside the docs directory so MkDocs can
    read the User Manual content directly from the canonical source in
    ``spectroview/resources/``.  The symlinks are removed automatically in
    :func:`on_post_build` so they never linger in the working tree.

    On platforms where symlinks are not supported (or permission-restricted),
    the hook falls back to a full directory copy.
    """
    docs_dir = config['docs_dir']
    project_root = os.path.dirname(docs_dir)  # one level above docs/

    # Source paths (canonical, inside the spectroview package)
    src_manual_dir = os.path.join(project_root,
                                  "spectroview", "resources", "user_manual")
    src_img_dir = os.path.join(project_root,
                               "spectroview", "resources",
                               "user_manual_images")

    # Destination paths (inside the MkDocs docs folder)
    dest_manual_dir = os.path.join(docs_dir, "user_manual")
    dest_img_dir = os.path.join(docs_dir, "user_manual_images")

    _ensure_link(src_manual_dir, dest_manual_dir, label="user manual")
    _ensure_link(src_img_dir, dest_img_dir, label="user manual images")


def on_post_build(config, **kwargs):
    """
    Hook that runs after the MkDocs build (or gh-deploy) finishes.

    Removes the temporary symlinks (or copied directories) that were created
    in :func:`on_pre_build`, keeping the working tree clean.
    """
    for path in _created_symlinks:
        if os.path.islink(path):
            os.unlink(path)
            print(f"  Cleaned up symlink: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  Cleaned up copied dir: {path}")
    _created_symlinks.clear()


# ---------------------------------------------------------------------- #
# Helper
# ---------------------------------------------------------------------- #

def _ensure_link(src: str, dest: str, *, label: str = "") -> None:
    """Create a directory symlink *dest* → *src*.

    If *dest* already exists as a correct symlink nothing is re-created.
    If it exists as a real directory (e.g. leftover from a previous copy-based
    workflow) it is removed first.
    Falls back to ``shutil.copytree`` when ``os.symlink`` is unavailable.
    """
    if not os.path.isdir(src):
        print(f"  Warning: source {label} directory not found at {src}")
        return

    # Remove stale destination (real dir or broken/wrong symlink)
    if os.path.islink(dest):
        if os.readlink(dest) == src:
            # Already correct – still register for cleanup
            _created_symlinks.append(dest)
            return
        os.unlink(dest)
    elif os.path.isdir(dest):
        shutil.rmtree(dest)

    # Try creating a symlink; fall back to a copy on failure
    try:
        os.symlink(src, dest, target_is_directory=True)
        print(f"  Linked {label}: {dest} -> {src}")
    except (OSError, NotImplementedError):
        shutil.copytree(src, dest)
        print(f"  Copied {label}: {src} -> {dest}  (symlink unavailable)")

    _created_symlinks.append(dest)
