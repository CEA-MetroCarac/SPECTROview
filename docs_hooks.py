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
    src_path = page.file.src_path.replace('\\', '/')
    if src_path.startswith("user_manual/"):
        # Determine if we are at the user_manual/ root (index.md)
        # or in a sub-directory (e.g. user_manual/01_introduction/)
        is_index = src_path.endswith("index.md")
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
    
    It also dynamically fetches GitHub releases and writes them to docs/changelog.md
    """
    docs_dir = config['docs_dir']
    
    # --- GitHub Releases Sync ---
    try:
        import urllib.request
        import json
        import ssl
        import os
        changelog_path = os.path.join(docs_dir, "changelog.md")
        context = ssl._create_unverified_context()
        req = urllib.request.Request("https://api.github.com/repos/CEA-MetroCarac/SPECTROview/releases")
        req.add_header('User-Agent', 'MkDocs-Hook')
        with urllib.request.urlopen(req, context=context, timeout=10) as response:
            releases = json.loads(response.read().decode())
            
        new_markdown = "# Changelog\n\n*(Dynamically synchronized from [GitHub Releases](https://github.com/CEA-MetroCarac/SPECTROview/releases))*\n\n---\n\n"
        for release in releases:
            tag = release.get("tag_name", "Unknown")
            name = release.get("name", tag)
            date = release.get("published_at", "")[:10]
            url = release.get("html_url", "")
            body = release.get("body", "")
            if body is None: body = ""
            
            # Convert raw <img src="..."> tags to Markdown format
            # This ensures MkDocs properly styles, lazy-loads, and displays them.
            import re
            def fix_img(m):
                src = m.group(1).replace("\\", "/")
                return f"![image]({src})"
            body = re.sub(r'<img[^>]*?src=[\'"]([^\'"]+)[\'"][^>]*?>', fix_img, body)
            
            # Demote headers to keep TOC clean (demote to h5 so they bypass toc_depth: 4)
            body = body.replace("\n### ", "\n##### ")
            body = body.replace("\n## ", "\n##### ")
            body = body.replace("\n# ", "\n##### ")
            if body.startswith("### "): body = "##### " + body[4:]
            elif body.startswith("## "): body = "##### " + body[3:]
            elif body.startswith("# "): body = "##### " + body[2:]
            
            # Prevent text immediately preceding '---' from becoming an H2
            body = re.sub(r'\n-{3,}', '\n\n---', body)
            
            new_markdown += f"## [{name}]({url}) - {date}\n\n{body}\n\n---\n\n"
            
        with open(changelog_path, 'w', encoding='utf-8') as f:
            f.write(new_markdown)
    except Exception as e:
        print(f"Failed to fetch GitHub releases: {e}")
    # ---------------------------

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


def on_files(files, config):
    """
    Hook that runs after MkDocs has scanned the docs directory.
    
    Since MkDocs 1.5, files matched by .gitignore are automatically excluded.
    Because docs/user_manual_images is in .gitignore, MkDocs ignores it.
    This hook manually re-adds the images to the MkDocs files collection
    so they are copied to the site/ output directory and link validation works.
    """
    from mkdocs.structure.files import File
    
    docs_dir = config['docs_dir']
    site_dir = config['site_dir']
    use_directory_urls = config.get('use_directory_urls', True)
    
    img_dir = os.path.join(docs_dir, 'user_manual_images')
    if os.path.isdir(img_dir):
        for root, _, filenames in os.walk(img_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, docs_dir).replace('\\', '/')
                
                # Only add if it wasn't already picked up
                if not any(f.src_uri == rel_path for f in files):
                    file_obj = File(rel_path, docs_dir, site_dir, use_directory_urls)
                    files.append(file_obj)
                    
    return files


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
        try:
            os.unlink(dest)
        except OSError:
            pass
    elif os.path.isdir(dest):
        try:
            shutil.rmtree(dest)
        except OSError:
            pass  # File lock from another mkdocs process, we will overwrite

    # Try creating a symlink; fall back to a copy on failure
    try:
        if not os.path.isdir(dest):
            os.symlink(src, dest, target_is_directory=True)
            print(f"  Linked {label}: {dest} -> {src}")
        else:
            raise OSError("Directory exists and is locked")
    except (OSError, NotImplementedError):
        shutil.copytree(src, dest, dirs_exist_ok=True)
        print(f"  Copied {label}: {src} -> {dest}  (symlink unavailable)")

    _created_symlinks.append(dest)
