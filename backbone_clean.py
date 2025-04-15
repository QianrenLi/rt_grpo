import os
import shutil
from pathlib import Path

def should_exclude(item):
    exclude = {'.git', '.gitmodules', 'README.md', 'LICENSE'}
    return item in exclude

def get_backbone_files():
    backbone_dir = "./backbone"
    if not os.path.exists(backbone_dir):
        raise FileNotFoundError(f"Backbone directory '{backbone_dir}' not found")
    
    return [item for item in os.listdir(backbone_dir) if not should_exclude(item)]

def clean_files():
    try:
        backbone_files = get_backbone_files()
    except FileNotFoundError as e:
        print(e)
        return
    
    print("Cleaning up files...")
    for item in backbone_files:
        target = os.path.join(".", item)
        if os.path.exists(target):
            if os.path.isdir(target):
                shutil.rmtree(target)
                print(f"Removed directory: {target}")
            else:
                os.remove(target)
                print(f"Removed file: {target}")
    
    # Check if there are any .rej or .orig files from patch
    for patch_artifact in Path(".").glob("*.rej"):
        patch_artifact.unlink()
        print(f"Removed patch rejection file: {patch_artifact}")
    for patch_artifact in Path(".").glob("*.orig"):
        patch_artifact.unlink()
        print(f"Removed patch backup file: {patch_artifact}")
    
    print("Cleanup completed.")

if __name__ == "__main__":
    clean_files()