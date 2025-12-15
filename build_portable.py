import os
import shutil
import subprocess
import sys
import time

def run_command(cmd, shell=True):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=shell)

def main():
    print("=== Building Portable AutoAnnotation Distribution ===")
    
    # 1. Configuration
    env_name = "phonetic_311"
    output_dir = "dist_portable"
    env_output_archive = os.path.join(output_dir, "env.tar.gz")
    env_extract_dir = os.path.join(output_dir, "env")
    app_dir = os.path.join(output_dir, "app")
    
    # 2. Prepare Directories
    if os.path.exists(output_dir):
        print(f"Cleaning previous build at {output_dir}...")
        # shutil.rmtree(output_dir) # Be careful, maybe just warn?
        # For now, let's keep it but remove subdirs if they exist to be fresh
        pass
    else:
        os.makedirs(output_dir)

    # 3. Pack Conda Environment
    if not os.path.exists(env_extract_dir):
        print(f"Packing conda environment '{env_name}'...")
        # We use the base conda to pack the target env
        # Note: --ignore-missing-files helps with some broken symlinks or cache files
        cmd = f"conda pack -n {env_name} -o \"{env_output_archive}\" --ignore-missing-files --n-threads 4"
        run_command(cmd)
        
        print(f"Extracting environment to {env_extract_dir}...")
        os.makedirs(env_extract_dir, exist_ok=True)
        
        # Use tar to extract (Windows 10+ has tar)
        # -xf: extract file, -C: change directory
        cmd = f"tar -xf \"{env_output_archive}\" -C \"{env_extract_dir}\""
        run_command(cmd)
        
        # Clean up archive to save space
        print("Removing archive...")
        os.remove(env_output_archive)
    else:
        print(f"Environment already exists at {env_extract_dir}, skipping packing (delete folder to force rebuild).")

    # 4. Copy App Files
    print(f"Copying application files to {app_dir}...")
    if os.path.exists(app_dir):
        shutil.rmtree(app_dir)
    os.makedirs(app_dir)
    
    # Files to copy
    files = [
        "auto_annotation_main.py",
        "rename_tool.py",
        "PhoneticToolbox.ico",
    ]
    dirs = [
        "views",
        "utils",
    ]
    
    for f in files:
        if os.path.exists(f):
            shutil.copy2(f, app_dir)
            print(f"Copied {f}")
            
    for d in dirs:
        if os.path.exists(d):
            shutil.copytree(d, os.path.join(app_dir, d))
            print(f"Copied directory {d}")

    # 5. Create Launcher Script
    print("Creating launcher script...")
    launcher_path = os.path.join(output_dir, "双击启动 (Start).bat")
    with open(launcher_path, "w", encoding="gbk") as f:
        f.write("@echo off\n")
        f.write("set \"SCRIPT_DIR=%~dp0\"\n")
        f.write("set \"ENV_DIR=%SCRIPT_DIR%env\"\n")
        f.write("set \"APP_DIR=%SCRIPT_DIR%app\"\n")
        f.write("\n")
        f.write("echo Initializing Environment... Please wait...\n")
        f.write("call \"%ENV_DIR%\\Scripts\\activate.bat\"\n")
        f.write("\n")
        f.write("echo Starting AutoAnnotation...\n")
        # Use pythonw.exe to avoid keeping a black console window (optional), 
        # but user might want to see logs. Let's use python.exe for now for debug safety.
        f.write("python \"%APP_DIR%\\auto_annotation_main.py\"\n")
        f.write("\n")
        f.write("if %ERRORLEVEL% NEQ 0 (\n")
        f.write("    echo Application exited with error code %ERRORLEVEL%\n")
        f.write("    pause\n")
        f.write(")\n")
    
    print("=== Build Complete! ===")
    print(f"Output Directory: {os.path.abspath(output_dir)}")
    print(f"You can zip the '{output_dir}' folder and send it to the user.")

if __name__ == "__main__":
    main()
