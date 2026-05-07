import os
import shutil

def on_pre_build(config, **kwargs):
    """
    Hook that runs before the MkDocs build process starts.
    It automatically copies the application's built-in User Manual section
    files and images into the MkDocs documentation directory, ensuring the
    website is always 100% in sync with the application's internal manual.
    """
    docs_dir = config['docs_dir']

    # Source paths (from the spectroview package)
    src_manual_dir = "spectroview/resources/user_manual"
    src_img = "spectroview/resources/user_manual_images"

    # Destination paths (in the MkDocs docs folder)
    dest_manual_dir = os.path.join(docs_dir, "user-manual")
    dest_img = os.path.join(docs_dir, "user_manual_images")

    print(f"Synchronizing user manual sections to: {dest_manual_dir}")

    # Copy all section markdown files
    if os.path.isdir(src_manual_dir):
        if os.path.exists(dest_manual_dir):
            shutil.rmtree(dest_manual_dir)
        shutil.copytree(src_manual_dir, dest_manual_dir)

        # Fix image paths: section files use ../user_manual_images/
        # but MkDocs expects paths relative to docs/ root.
        # Since dest is docs/user-manual/ and images are at
        # docs/user_manual_images/, the ../user_manual_images/ path
        # resolves correctly. No rewriting needed.
    else:
        print(f"Warning: Source manual directory not found at {src_manual_dir}")

    # Copy images folder to docs/user_manual_images/
    if os.path.isdir(src_img):
        if os.path.exists(dest_img):
            shutil.rmtree(dest_img)
        shutil.copytree(src_img, dest_img)
    else:
        print(f"Warning: Source images not found at {src_img}")
