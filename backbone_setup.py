import os
import shutil
import subprocess
from pathlib import Path

def should_exclude(item):
    exclude = {'.git', '.gitmodules', 'README.md', 'LICENSE'}
    return item in exclude

def copy_backbone_files():
    backbone_dir = "./backbone"
    if not os.path.exists(backbone_dir):
        raise FileNotFoundError(f"Backbone directory '{backbone_dir}' not found")
    
    print("Copying files from backbone...")
    for item in os.listdir(backbone_dir):
        if should_exclude(item):
            print(f"Skipping excluded item: {item}")
            continue
            
        src = os.path.join(backbone_dir, item)
        dst = os.path.join(".", item)
        
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('.git', '.gitmodules'))
        else:
            shutil.copy2(src, dst)
    print("Files copied successfully.")

def apply_patch():
    patch_files = ["env_patch.patch", "param.patch"]
    for patch_file in patch_files:
        if not os.path.exists(patch_file):
            raise FileNotFoundError(f"Patch file '{patch_file}' not found")
        
        print(f"Applying patch {patch_file}...")
        try:
            subprocess.run(["git", "apply", patch_file], check=True)
            print("Patch applied successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to apply patch: {e}")
        except FileNotFoundError:
            print("Git not found. Please install Git to apply patches.")

def main():
    try:
        copy_backbone_files()
        apply_patch()
    except Exception as e:
        print(f"Setup failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()