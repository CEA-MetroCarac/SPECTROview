import os
import shutil
import re


# Keep track of symlinks created so on_post_build can clean them up.
_created_symlinks: list[str] = []


def on_page_content(html, page, config, files):
    """
    Hook to fix raw HTML links in the generated HTML content.
    
    MkDocs normally doesn't rewrite links inside raw HTML <a> tags.
    This hook finds relative href="filename.md" links in the user manual
    and rewrites them to href="filename/" so they match MkDocs' 
    directory-style URL structure.
    """
    if page.file.src_path.startswith("user_manual/"):
        # Determine if we are at the user_manual/ root (index.md)
        # or in a sub-directory (e.g. user_manual/01_introduction/)
        is_index = page.file.src_path.endswith("index.md")
        prefix = "" if is_index else "../"

        def replace_link(match):
            filename = match.group(1)
            anchor = match.group(2)
            # Special case for index.md: link to the directory root
            if filename == "index":
                return f'href="{prefix or "./"}{anchor}"'
            # General case: link to the file's directory URL
            return f'href="{prefix}{filename}/{anchor}"'

        # Match href="filename.md" or href="filename.md#anchor"
        # but ignore external (http) or absolute (/) links.
        html = re.sub(
            r'href="(?!(?:https?://|/))([^"#]+)\.md(#?[^"]*)"',
            replace_link,
            html
        )

        # Fix image src attributes inside HTML tags.
        # MkDocs automatically fixes relative paths for Markdown images ![alt](path),
        # but leaves raw HTML <img src="..."> untouched.
        # Images are symlinked to site/user_manual_images/
        # From site/user_manual/index.html (is_index=True), "../user_manual_images/" is correct.
        # From site/user_manual/page_name/index.html (is_index=False), we need "../../user_manual_images/"
        if not is_index:
            html = re.sub(
                r'src="\.\./user_manual_images/',
                r'src="../../user_manual_images/',
                html
            )
    return html


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

def on_page_markdown(markdown, page, config, files):
    """
    Hook to dynamically fetch GitHub releases for the changelog page.
    """
    if page.file.src_path == "changelog.md":
        import urllib.request
        import json
        import ssl
        try:
            context = ssl._create_unverified_context()
            req = urllib.request.Request("https://api.github.com/repos/CEA-MetroCarac/SPECTROview/releases")
            req.add_header('User-Agent', 'MkDocs-Hook')
            with urllib.request.urlopen(req, context=context, timeout=10) as response:
                releases = json.loads(response.read().decode())
                
            new_markdown = "# Changelog\n\n"
            new_markdown += "*(Dynamically synchronized from [GitHub Releases](https://github.com/CEA-MetroCarac/SPECTROview/releases))*\n\n---\n\n"
            for release in releases:
                tag = release.get("tag_name", "Unknown")
                name = release.get("name", tag)
                date = release.get("published_at", "")[:10]
                url = release.get("html_url", "")
                body = release.get("body", "")
                
                new_markdown += f"## [{name}]({url}) - {date}\n\n{body}\n\n---\n\n"
            return new_markdown
        except Exception as e:
            print(f"Failed to fetch GitHub releases: {e}")
            return markdown
    return markdown
