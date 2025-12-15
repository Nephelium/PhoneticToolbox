import os
import shutil
import json
import sys
from pypinyin import lazy_pinyin

LOG_FILE = "rename_log.json"

def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'

def has_chinese(text):
    return any(is_chinese(char) for char in text)

def to_pinyin(text):
    result = []
    for char in text:
        if is_chinese(char):
            result.extend(lazy_pinyin(char))
        else:
            result.append(char)
    # Join and replace spaces with underscores for safety
    return "".join(result).replace(" ", "_")

def batch_rename_process(target_dir, mode='1'):
    """
    Execute the batch rename process.
    mode: '1' for overwrite, '2' for save as new folder (suffix _en)
    Returns: (success, new_work_dir, log_path, message)
    """
    if not os.path.exists(target_dir):
        return False, None, None, "错误: 文件夹不存在 (Error: Folder not found)"

    work_dir = target_dir
    
    if mode == '2':
        parent_dir = os.path.dirname(target_dir)
        dir_name = os.path.basename(target_dir)
        new_dir_name = dir_name + "_en"
        work_dir = os.path.join(parent_dir, new_dir_name)
        
        try:
            if os.path.exists(work_dir):
                return False, None, None, f"错误: 目标文件夹已存在: {work_dir}"
            shutil.copytree(target_dir, work_dir)
        except Exception as e:
            return False, None, None, f"复制失败: {e}"
    elif mode != '1':
        return False, None, None, "无效选择 (Invalid choice)"

    # Process
    operations = []
    count = 0
    
    # Walk bottom-up to handle children before parents
    for root, dirs, files in os.walk(work_dir, topdown=False):
        # Rename Files
        for filename in files:
            if has_chinese(filename):
                old_path = os.path.join(root, filename)
                name_part, ext_part = os.path.splitext(filename)
                new_name = to_pinyin(name_part) + ext_part
                
                try:
                    # Rename
                    new_full_path = os.path.join(root, new_name)
                    if old_path != new_full_path:
                        os.rename(old_path, new_full_path)
                        operations.append({'old': old_path, 'new': new_full_path})
                        count += 1
                        
                        # --- Feature Request: Auto-infer TextGrid mapping ---
                        # Even if TextGrid doesn't exist yet, we predict its path for future restoration.
                        # Check if this file is an audio file
                        if ext_part.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.aiff']:
                            # Predict TextGrid paths
                            old_tg = os.path.splitext(old_path)[0] + ".TextGrid"
                            new_tg = os.path.splitext(new_full_path)[0] + ".TextGrid"
                            
                            # Add to operations list so it can be "restored" (renamed back) later
                            # Note: We don't check os.path.exists(new_tg) here because it is created later by MFA.
                            # But for restoration to work, we just need to know: if new_tg exists, rename it to old_tg.
                            operations.append({'old': old_tg, 'new': new_tg})
                        # ----------------------------------------------------
                        
                except Exception as e:
                    print(f"Failed to rename {filename}: {e}")

        # Rename Directories
        for dirname in dirs:
            if has_chinese(dirname):
                old_path = os.path.join(root, dirname)
                new_dirname = to_pinyin(dirname)
                
                try:
                    new_full_path = os.path.join(root, new_dirname)
                    if old_path != new_full_path:
                        os.rename(old_path, new_full_path)
                        operations.append({'old': old_path, 'new': new_full_path})
                        count += 1
                except Exception as e:
                    print(f"Failed to rename directory {dirname}: {e}")

    # Save Log
    log_path = os.path.join(work_dir, LOG_FILE)
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(operations, f, indent=4, ensure_ascii=False)
    except Exception as e:
        return False, work_dir, None, f"保存日志失败: {e}"
        
    return True, work_dir, log_path, f"完成! 共重命名 {count} 个项目。"

def restore_process(log_path):
    """
    Restore names from log file.
    Returns: (success, message)
    """
    if not os.path.exists(log_path):
        return False, "错误: 找不到日志路径 (Error: Path not found)"
        
    real_log_path = log_path
    if os.path.isdir(log_path):
        real_log_path = os.path.join(log_path, LOG_FILE)
        if not os.path.exists(real_log_path):
             return False, f"错误: 在该目录下找不到日志文件 {LOG_FILE}"

    try:
        with open(real_log_path, 'r', encoding='utf-8') as f:
            operations = json.load(f)
    except Exception as e:
        return False, f"读取日志失败: {e}"
        
    success_count = 0
    # Reverse operations to restore parents before children?
    # Renaming was: Child renamed, then Parent renamed.
    # Log order: Child Op, Parent Op.
    # To restore:
    # 1. Restore Parent (Parent Op is last in list).
    # 2. Restore Child (Child Op is earlier).
    # So we need to process the list in REVERSE order.
    
    for op in reversed(operations):
        new_path = op['new']
        old_path = op['old']
        
        try:
            if os.path.exists(new_path):
                os.rename(new_path, old_path)
                success_count += 1
            else:
                print(f"Warning: Path not found {new_path}")
        except Exception as e:
            print(f"Failed to restore {new_path}: {e}")
            
    return True, f"恢复完成! 成功恢复 {success_count}/{len(operations)} 个项目。"

def start_renaming():
    print("=== 中文文件名转英文(拼音)工具 ===")
    print("=== Chinese to English (Pinyin) Renaming Tool ===")
    
    # 1. Get Directory
    target_dir = input("请输入要处理的文件夹路径 (Enter folder path): ").strip()
    # Remove quotes if user dragged and dropped
    if (target_dir.startswith('"') and target_dir.endswith('"')) or \
       (target_dir.startswith("'") and target_dir.endswith("'")):
        target_dir = target_dir[1:-1]
        
    if not os.path.exists(target_dir):
        print("错误: 文件夹不存在 (Error: Folder not found)")
        return

    # 2. Choose Mode
    print("\n请选择模式 (Choose Mode):")
    print("1. 覆盖原文件夹 (Overwrite / Rename in place)")
    print("2. 另存为新文件夹 (Save as new folder)")
    choice = input("请输入 1 或 2: ").strip()
    
    work_dir = target_dir
    
    if choice == '2':
        parent_dir = os.path.dirname(target_dir)
        dir_name = os.path.basename(target_dir)
        new_dir_name = dir_name + "_en"
        work_dir = os.path.join(parent_dir, new_dir_name)
        
        # Ask for custom output name? Or just auto? Auto is easier for now.
        print(f"正在复制文件到新文件夹: {work_dir} ...")
        try:
            if os.path.exists(work_dir):
                print("错误: 目标文件夹已存在，请先删除或重命名。")
                return
            shutil.copytree(target_dir, work_dir)
        except Exception as e:
            print(f"复制失败: {e}")
            return
    elif choice != '1':
        print("无效选择 (Invalid choice)")
        return

    # 3. Process
    print("\n开始重命名 (Starting rename)...")
    operations = []
    count = 0
    
    # Walk bottom-up to handle children before parents
    for root, dirs, files in os.walk(work_dir, topdown=False):
        # Rename Files
        for filename in files:
            if has_chinese(filename):
                old_path = os.path.join(root, filename)
                name_part, ext_part = os.path.splitext(filename)
                new_name = to_pinyin(name_part) + ext_part
                
                try:
                    # Rename
                    new_full_path = os.path.join(root, new_name)
                    if old_path != new_full_path:
                        os.rename(old_path, new_full_path)
                        operations.append({'old': old_path, 'new': new_full_path})
                        print(f"[File] {filename} -> {new_name}")
                        count += 1
                except Exception as e:
                    print(f"Failed to rename {filename}: {e}")

        # Rename Directories
        for dirname in dirs:
            if has_chinese(dirname):
                old_path = os.path.join(root, dirname)
                new_dirname = to_pinyin(dirname)
                
                try:
                    new_full_path = os.path.join(root, new_dirname)
                    if old_path != new_full_path:
                        os.rename(old_path, new_full_path)
                        operations.append({'old': old_path, 'new': new_full_path})
                        print(f"[Dir]  {dirname} -> {new_dirname}")
                        count += 1
                except Exception as e:
                    print(f"Failed to rename directory {dirname}: {e}")

    # 4. Save Log
    log_path = os.path.join(work_dir, LOG_FILE)
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(operations, f, indent=4, ensure_ascii=False)
        print(f"\n完成! 共重命名 {count} 个项目。")
        print(f"恢复记录已保存至: {log_path}")
        print("如果需要恢复，请运行此脚本并选择恢复模式(尚未实现，请手动调用 restore 函数)。")
        # To make it user friendly, I'll update main to ask mode.
    except Exception as e:
        print(f"保存日志失败: {e}")

def restore_names():
    print("=== 恢复文件名 (Restore Names) ===")
    log_path = input("请输入 rename_log.json 文件的路径: ").strip()
    if (log_path.startswith('"') and log_path.endswith('"')) or \
       (log_path.startswith("'") and log_path.endswith("'")):
        log_path = log_path[1:-1]
        
    if not os.path.exists(log_path):
        print("错误: 找不到路径 (Error: Path not found)")
        return
        
    # If user provided a directory, append the default log filename
    if os.path.isdir(log_path):
        log_path = os.path.join(log_path, LOG_FILE)
        if not os.path.exists(log_path):
             print(f"错误: 在该目录下找不到日志文件 {LOG_FILE} (Error: Log file not found in directory)")
             return

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            operations = json.load(f)
    except Exception as e:
        print(f"读取日志失败: {e}")
        return
        
    print(f"找到 {len(operations)} 条重命名记录。开始恢复...")
    
    # Reverse operations to restore parents before children?
    # Renaming was: Child renamed, then Parent renamed.
    # Log order: Child Op, Parent Op.
    # To restore:
    # 1. Restore Parent (Parent Op is last in list).
    # 2. Restore Child (Child Op is earlier).
    # So we need to process the list in REVERSE order.
    
    success_count = 0
    for op in reversed(operations):
        new_path = op['new']
        old_path = op['old']
        
        try:
            if os.path.exists(new_path):
                os.rename(new_path, old_path)
                print(f"Restored: {os.path.basename(new_path)} -> {os.path.basename(old_path)}")
                success_count += 1
            else:
                print(f"Warning: Path not found {new_path}")
        except Exception as e:
            print(f"Failed to restore {new_path}: {e}")
            
    print(f"\n恢复完成! 成功恢复 {success_count}/{len(operations)} 个项目。")

if __name__ == "__main__":
    print("1. 批量重命名 (中文 -> 英文/拼音)")
    print("2. 恢复文件名 (英文 -> 中文)")
    choice = input("请选择功能 (1/2): ").strip()
    
    if choice == '1':
        start_renaming()
    elif choice == '2':
        restore_names()
    else:
        print("无效输入")
