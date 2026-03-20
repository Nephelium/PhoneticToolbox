import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import subprocess
import tempfile
import sys
import warnings
from scipy.io import wavfile
from scipy import signal

def _find_reaper_bin(explicit: Optional[str] = None) -> str:
    """查找 REAPER 可执行文件。"""
    def _resolve_candidate(path_like: str | Path) -> str | None:
        try:
            p = Path(path_like)
            if p.is_file():
                return str(p.resolve())
            if p.is_dir():
                exe = p / "reaper.exe"
                if exe.is_file():
                    return str(exe.resolve())
                exe2 = p / "reaper"
                if exe2.is_file():
                    return str(exe2.resolve())
        except Exception:
            return None
        return None

    if explicit:
        r = _resolve_candidate(explicit)
        if r:
            return r
        w = shutil.which(str(explicit))
        if w:
            return w

    for name in ("reaper", "reaper.exe"):
        w = shutil.which(name)
        if w:
            return w

    # 启发式搜索
    # 假设当前结构: phonetic_toolbox/core/acoustic/f0_reaper.py
    # 向上寻找项目根目录
    cands = [
        Path.cwd() / "reaper.exe",
        Path.cwd() / "reaper" / "reaper.exe",
        Path.cwd() / "bin" / "reaper.exe",
        Path(__file__).parent.parent.parent.parent / "bin" / "reaper.exe",
    ]
    
    for cand in cands:
        r = _resolve_candidate(cand)
        if r:
            return r
            
    raise RuntimeError("REAPER binary not found.")

def _parse_reaper_est_f0(path: Path) -> Tuple[List[float], List[int], List[float]]:
    """解析 REAPER 输出文件。"""
    times = []
    voiced = []
    values = []
    try:
        with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
            in_header = True
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if in_header:
                    if s == "EST_Header_End":
                        in_header = False
                    continue
                parts = s.split()
                if len(parts) < 3:
                    continue
                try:
                    t = float(parts[0])
                    v = int(float(parts[1]))
                    val = float(parts[2])
                except Exception:
                    continue
                times.append(t)
                voiced.append(v)
                values.append(val)
    except Exception:
        pass
    return times, voiced, values

def compute_reaper_f0(
    wav_path: Path,
    frame_interval_sec: float,
    min_f0: float,
    max_f0: float,
    hilbert: bool = False,
    no_highpass: bool = False,
    reaper_bin: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    使用 REAPER 算法计算 F0。

    Args:
        wav_path (Path): 音频文件路径。
        frame_interval_sec (float): 帧间隔 (秒)。
        min_f0 (float): 最小 F0。
        max_f0 (float): 最大 F0。
        hilbert (bool): 是否使用 Hilbert 变换。
        no_highpass (bool): 是否禁用高通滤波。
        reaper_bin (Optional[str]): REAPER 可执行文件路径。

    Returns:
        Dict[str, np.ndarray]: 包含 "rTimes", "rVoiced", "rF0Raw", "rF0" 的字典。
    """
    try:
        # Try to find REAPER binary
        rb = _find_reaper_bin(reaper_bin)
        use_binary = True
    except RuntimeError as e:
        print(f"REAPER binary not found: {e}. Will try Python implementation.")
        use_binary = False
    except Exception as e:
        print(f"Unexpected error finding REAPER binary: {e}. Will try Python implementation.")
        use_binary = False
    
    tmpdir = None
    try:
        tmpdir = tempfile.TemporaryDirectory()
        converted_wav = Path(tmpdir.name) / "input_16k.wav"
        
        # 转换为 16k mono int16 以兼容 REAPER
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", wavfile.WavFileWarning)
                fs, data = wavfile.read(wav_path)
            
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            
            target_fs = 16000
            if fs != target_fs:
                num_samples = int(len(data) * target_fs / fs)
                data = signal.resample(data, num_samples)
                fs = target_fs
                
            if data.dtype.kind == 'f':
                max_val = np.max(np.abs(data))
                if max_val > 0:
                    data = data / max_val * 32000.0
                data = data.astype(np.int16)
            elif data.dtype == np.int32:
                 data = (data / 65536.0).astype(np.int16)
            elif data.dtype == np.uint8:
                 data = ((data.astype(float) - 128.0) * 256.0).astype(np.int16)
            
            if data.dtype != np.int16:
                data = data.astype(np.int16)
                
            wavfile.write(converted_wav, 16000, data)
            process_wav_path = converted_wav
        except Exception as e:
            print(f"Warning: Failed to convert audio for REAPER: {e}. Using original file.")
            process_wav_path = wav_path

        times, voiced, values = [], [], []
        binary_success = False

        if use_binary:
            f0_out = Path(tmpdir.name) / "out.f0"
            cmd = [
                rb,
                "-i", str(process_wav_path),
                "-f", str(f0_out),
                "-a",
                "-e", str(float(frame_interval_sec)),
                "-m", str(float(min_f0)),
                "-x", str(float(max_f0)),
            ]
            if hilbert:
                cmd.append("-t")
            if no_highpass:
                cmd.append("-s")
            
            startupinfo = None
            creationflags = 0
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                creationflags = subprocess.CREATE_NO_WINDOW
            
            try:
                r = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    startupinfo=startupinfo,
                    creationflags=creationflags
                )
                if r.returncode != 0:
                    err_msg = r.stderr.decode(errors="ignore")
                    # print(f"REAPER binary failed with return code {r.returncode}")
                    # print(f"REAPER stderr: {err_msg}")
                    raise RuntimeError(err_msg or "reaper failed")
                
                times, voiced, values = _parse_reaper_est_f0(f0_out)
                binary_success = True
            except Exception as e:
                print(f"REAPER binary failed: {e}. Falling back to Python implementation.")
                binary_success = False

        if not binary_success:
            try:
                from . import reaper_python
                times, voiced, values = reaper_python.run_python_impl(
                    str(process_wav_path), 
                    frame_interval_sec, 
                    min_f0, 
                    max_f0, 
                    hilbert, 
                    no_highpass
                )
            except Exception as e:
                print(f"REAPER Python implementation failed: {e}")
                raise e

        t_arr = np.array(times, dtype=float)
        v_arr = np.array(voiced, dtype=int)
        val_arr = np.array(values, dtype=float)
        out_series = np.where(val_arr > 0.0, val_arr, np.nan)
        return {"rTimes": t_arr, "rVoiced": v_arr, "rF0Raw": val_arr, "rF0": out_series}
    except Exception as e:
        print(f"REAPER F0 computation error: {e}")
        return {"rTimes": np.array([]), "rVoiced": np.array([]), "rF0Raw": np.array([]), "rF0": np.array([])}
    finally:
        if tmpdir is not None:
            try:
                tmpdir.cleanup()
            except Exception:
                pass
