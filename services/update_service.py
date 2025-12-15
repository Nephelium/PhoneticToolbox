import json
import sys
import os
import subprocess
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Tuple, Optional
from packaging import version

# =========================================================================
# 配置区域
GITHUB_REPO = "Nephelium/PhoneticToolbox"
VERSION_CHECK_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
# =========================================================================

def get_current_version() -> str:
    try:
        from PhoneticToolbox import __version__
        return __version__
    except ImportError:
        return "1.0.0"

def check_for_updates(current_version_str: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    检查更新 (GitHub Releases)
    Returns: (has_update, latest_version, download_url, changelog)
    """
    try:
        # 添加 User-Agent 防止 GitHub API 403 Forbidden
        req = urllib.request.Request(
            VERSION_CHECK_URL, 
            headers={'User-Agent': 'PhoneticToolbox-Updater'}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status != 200:
                return False, None, None, None
            
            data = json.loads(response.read().decode('utf-8'))
            
            # 解析 GitHub Release JSON
            tag_name = data.get("tag_name", "0.0.0")
            latest_version_str = tag_name.lstrip('v') # 去除 'v' 前缀
            changelog = data.get("body", "")
            assets = data.get("assets", [])
            
            # 寻找 EXE 下载链接
            download_url = None
            for asset in assets:
                name = asset.get("name", "")
                url = asset.get("browser_download_url", "")
                if name.lower() == "phonetictoolbox.exe":
                    download_url = url
                    break
                if name.lower().endswith(".exe"):
                    download_url = url
            
            # 如果没找到 exe，尝试取第一个 asset
            if not download_url and assets:
                download_url = assets[0].get("browser_download_url")

            # 使用 packaging.version 进行比较
            try:
                cur_v = version.parse(current_version_str)
                lat_v = version.parse(latest_version_str)
                if lat_v > cur_v:
                    return True, latest_version_str, download_url, changelog
            except Exception:
                # Fallback to string comparison
                if latest_version_str > current_version_str:
                    return True, latest_version_str, download_url, changelog
            
            # 即使没有更新，也返回最新版本号供界面显示 "已是最新版 (v1.0.0)"
            return False, latest_version_str, download_url, changelog

    except Exception as e:
        print(f"Check update failed: {e}")
        return False, None, None, None

def perform_update(download_url: str, progress_callback=None) -> None:
    """
    下载并执行更新
    """
    try:
        # 1. 确定下载路径
        temp_dir = tempfile.gettempdir()
        new_exe_name = "PhoneticToolbox_New.exe"
        new_exe_path = os.path.join(temp_dir, new_exe_name)
        
        # 2. 下载文件
        def report(blocknum, blocksize, totalsize):
            if progress_callback and totalsize > 0:
                percent = int(blocknum * blocksize * 100 / totalsize)
                progress_callback(min(percent, 100))
        
        print(f"Downloading from {download_url} to {new_exe_path}...")
        urllib.request.urlretrieve(download_url, new_exe_path, report)
        
        # 3. 生成更新脚本 (Bat)
        # 脚本逻辑：
        # a. 等待当前进程退出
        # b. 将旧 EXE 重命名/删除 (Windows下运行中不能删除，但通常可以重命名，或者等待结束)
        # c. 移动新 EXE 到原位置
        # d. 启动新 EXE
        # e. 删除脚本自身
        
        current_exe = sys.executable
        # 如果是 Python 脚本运行而非打包 EXE，sys.executable 是 python.exe，这会出问题。
        # 这里假设是打包后的环境。如果是开发环境，仅提示下载成功。
        
        if not getattr(sys, 'frozen', False):
            print("Running in source mode. Update downloaded but will not apply automatically.")
            os.startfile(temp_dir) # 打开文件夹
            return

        current_dir = os.path.dirname(current_exe)
        exe_name = os.path.basename(current_exe)
        bat_path = os.path.join(temp_dir, "update_phonetic_toolbox.bat")
        
        # 写入 Bat 脚本
        # Ping 命令用于延时 (timeout 在某些旧系统可能不可用，ping 127.0.0.1 -n 3 约等于 2秒)
        bat_content = f"""
@echo off
echo Waiting for application to close...
ping 127.0.0.1 -n 3 > nul

echo Updating files...
move /y "{new_exe_path}" "{current_exe}"

echo Restarting application...
start "" "{current_exe}"

del "%~f0"
"""
        with open(bat_path, "w") as f:
            f.write(bat_content)
            
        # 4. 运行脚本并退出
        print("Starting update script...")
        subprocess.Popen(bat_path, shell=True)
        sys.exit(0)
        
    except Exception as e:
        raise e
