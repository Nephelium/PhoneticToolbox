from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import csv
import os
import numpy as np


def _ensure_dir(p: Path) -> None:
    d = p.parent
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, data: Dict[str, Any]) -> None:
    _ensure_dir(path)
    flat_data: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat_data[f"{k}.{sk}"] = sv
        else:
            flat_data[k] = v
    max_len = 0
    for v in flat_data.values():
        if isinstance(v, (list, np.ndarray)):
            max_len = max(max_len, len(v))
    if max_len == 0:
        max_len = 1

    columns: Dict[str, List[str]] = {}
    original_keys: List[str] = list(flat_data.keys())

    def is_constant_value(val: Any) -> bool:
        if isinstance(val, (list, np.ndarray)):
            arr = np.array(val)
            if arr.size == 0:
                return False
            try:
                first = arr[0]
                return bool(np.all(arr == first))
            except Exception:
                return False
        return True

    special_last = ["TextGrid"]
    # Identify dynamic TextGrid columns (textgrid_*)
    tg_cols = [k for k in original_keys if k.lower().startswith("textgrid_")]
    special_last.extend(sorted(tg_cols))
    
    priority = [k for k in ["rTimes", "rF0", "pF0"] if k in original_keys]
    others = [k for k in original_keys if k not in priority and k not in special_last]
    non_constant = [k for k in others if not is_constant_value(flat_data[k])]
    constant = [k for k in others if is_constant_value(flat_data[k])]
    ordered_keys = priority + non_constant + constant
    for k in special_last:
        if k in original_keys:
            ordered_keys.append(k)

    for k in ordered_keys:
        v = flat_data[k]
        col_data: List[str] = []
        if isinstance(v, (list, np.ndarray)):
            arr = np.array(v)
            for i in range(max_len):
                if i < arr.size:
                    col_data.append(str(arr[i]))
                else:
                    col_data.append("")
        else:
            val_str = str(v)
            for i in range(max_len):
                col_data.append(val_str)
        columns[k] = col_data

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(ordered_keys)
        for i in range(max_len):
            w.writerow([columns[k][i] for k in ordered_keys])


def load_csv_any(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    
    try:
        with path.open("r", encoding="utf-8") as f:
            # Peek at the first line to detect format
            first_line = f.readline()
            f.seek(0)
            
            # 1. Legacy Long Format (key, index, value)
            if "key" in first_line and "index" in first_line and "value" in first_line:
                reader = csv.DictReader(f)
                buckets: Dict[str, List[Tuple[int, str]]] = {}
                scalars: Dict[str, str] = {}
                for row in reader:
                    key = row.get("key", "").strip()
                    idx = row.get("index", "").strip()
                    val = row.get("value", "")
                    if idx == "":
                        scalars[key] = val
                    else:
                        try:
                            i = int(idx)
                        except Exception:
                            i = 0
                        buckets.setdefault(key, []).append((i, val))
                for k, vals in buckets.items():
                    vals.sort(key=lambda x: x[0])
                    arr_vals: List[float] = []
                    is_float = True
                    for _, v in vals:
                        try:
                            arr_vals.append(float(v))
                        except Exception:
                            is_float = False
                            break
                    if is_float:
                        out[k] = np.array(arr_vals, dtype=float)
                    else:
                        out[k] = np.array([v for _, v in vals], dtype=object)
                for k, v in scalars.items():
                    try:
                        out[k] = float(v)
                    except Exception:
                        out[k] = v
                
                # Handle ManualData substructure if present in long format keys
                manual_keys = [kk for kk in out.keys() if kk.startswith("ManualData.")]
                if manual_keys:
                    md: Dict[str, Any] = {}
                    for mk in manual_keys:
                        sub = mk.split(".", 1)[1]
                        md[sub] = out.pop(mk)
                    out["ManualData"] = md
                return out

            # 2. Legacy Manual Data Format (Tab-separated with specific headers)
            header = first_line.strip().split("\t")
            if set(["参数", "时间_ms", "数值", "文件"]).issubset(set(header)):
                reader = csv.DictReader(f, delimiter="\t")
                params: List[str] = []
                times: List[float] = []
                values: List[float] = []
                files: List[str] = []
                for row in reader:
                    params.append(str(row.get("参数", "")))
                    try:
                        times.append(float(row.get("时间_ms", "")))
                    except Exception:
                        times.append(np.nan)
                    try:
                        values.append(float(row.get("数值", "")))
                    except Exception:
                        values.append(np.nan)
                    files.append(str(row.get("文件", "")))
                out["ManualData"] = {
                    "md_param": np.array(params, dtype=object),
                    "md_time_ms": np.array(times, dtype=float),
                    "md_value": np.array(values, dtype=float),
                    "md_file": np.array(files, dtype=object),
                }
                return out

            # 3. Wide Format (Default for new save_csv)
            # Assuming comma separated, first row is header
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return out
                
            data_cols: Dict[str, List[Any]] = {k: [] for k in reader.fieldnames}
            for row in reader:
                for k, v in row.items():
                    if k is not None: # DictReader might have None for extra fields
                        data_cols[k].append(v)
            
            # Convert lists to numpy arrays or scalars
            for k, vals in data_cols.items():
                # Check if all values are the same (scalar candidate)
                # But safer to always return array for consistency, 
                # or check if it's a parameter array.
                # However, original load_csv_any returned scalars for things like "frameshift".
                # If we saved frameshift as a constant column, we should ideally return it as a scalar
                # if the consumer expects a scalar.
                # But it's hard to know.
                # Let's try to convert to float array first.
                
                try:
                    # Replace empty strings with nan for float conversion
                    float_vals = []
                    for v in vals:
                        if v == "" or v is None:
                            float_vals.append(np.nan)
                        else:
                            float_vals.append(float(v))
                    arr = np.array(float_vals, dtype=float)
                    
                    # Heuristic: if all values are the same (and not all NaN), it might be a scalar?
                    # But for time-series, a constant parameter is also valid.
                    # The previous logic distinguished by "index" being empty.
                    # Here we don't have that info.
                    # Let's just return array. If the code expects scalar, it might fail or need adjustment.
                    # Wait, ManualDataController expects "frameshift" to be a scalar.
                    # "frameshift = float(data.get("frameshift", self.state.frameshift))"
                    # If data["frameshift"] is an array, float(arr) works only if size 1.
                    # If size > 1, it raises error.
                    # So we should detect constant columns?
                    
                    is_constant = False
                    if len(vals) > 0:
                        first = vals[0]
                        if all(x == first for x in vals):
                            is_constant = True
                    
                    if is_constant and len(vals) > 0:
                         # Try to return as scalar
                         out[k] = float_vals[0]
                    else:
                         out[k] = arr
                         
                except ValueError:
                    # String data
                    arr = np.array(vals, dtype=object)
                    is_constant = False
                    if len(vals) > 0:
                        first = vals[0]
                        if all(x == first for x in vals):
                            is_constant = True
                    if is_constant and len(vals) > 0:
                        out[k] = vals[0]
                    else:
                        out[k] = arr

            return out

    except Exception as e:
        print(f"Error loading CSV {path}: {e}")
        return out

