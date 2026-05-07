import os
import shutil


def on_pre_build(config, **kwargs):
    """
    Hook that runs before the MkDocs build process starts.

    Instead of copying the User Manual files into the docs directory, this
    hook creates symbolic links so MkDocs reads directly from the canonical
    source in ``spectroview/resources/``.  This guarantees the online
    documentation is always 100 % in sync with the application's built-in
    manual without maintaining a duplicate set of files.

    On platforms where symlinks are not supported (or permission-restricted),
    the hook falls back to a full copy.
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
    dest_manual_dir = os.path.join(docs_dir, "user-manual")
    dest_img_dir = os.path.join(docs_dir, "user_manual_images")

    _ensure_symlink(src_manual_dir, dest_manual_dir, label="user manual")
    _ensure_symlink(src_img_dir, dest_img_dir, label="user manual images")


def _ensure_symlink(src: str, dest: str, *, label: str = "") -> None:
    """Create a directory symlink *dest* → *src*.

    If *dest* already exists as a correct symlink nothing is done.
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
            print(f"  Symlink for {label} already up-to-date.")
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
