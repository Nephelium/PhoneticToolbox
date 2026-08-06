import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import sqlite3

FAST_PARAM_SUFFIX = ".ptb.sqlite"

def _is_fastdb_valid(path: Path) -> bool:
    if not path.exists():
        return False
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.execute(
            'SELECT COUNT(1), MIN(CAST("Time_s" AS REAL)), MAX(CAST("Time_s" AS REAL)) FROM params'
        )
        row = cur.fetchone()
        if row is None:
            return False
        cnt, tmin, tmax = row
        if cnt is None or int(cnt) <= 0:
            return False
        if tmin is None or tmax is None:
            return False
        return True
    except Exception:
        return False
    finally:
        conn.close()

def save_excel(path: Path, data: Dict[str, Any]) -> None:
    """
    Save dictionary data to an Excel file (.xlsx).
    Handles mixed scalar and array data by broadcasting scalars.
    """
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Determine max length
    max_len = 0
    for v in data.values():
        if isinstance(v, (list, np.ndarray)):
            max_len = max(max_len, len(v))
    
    if max_len == 0:
        max_len = 1

    # 2. Prepare DataFrame dict
    df_data = {}
    
    # Priority keys sorting
    original_keys = list(data.keys())
    
    # Identify dynamic TextGrid columns (textgrid_*)
    tg_cols = [k for k in original_keys if k.lower().startswith("textgrid_")]
    
    priority = [k for k in ["Time_s", "pF0", "rF0"] if k in original_keys]
    special_last = ["TextGrid"] + sorted(tg_cols)
    others = [k for k in original_keys if k not in priority and k not in special_last]
    
    ordered_keys = priority + others + special_last
    
    for k in ordered_keys:
        if k not in data:
            continue
        v = data[k]
        if isinstance(v, (list, np.ndarray)):
            # Handle array
            arr = np.array(v)
            if len(arr) < max_len:
                # Pad with NaNs or empty strings?
                # Pandas handles length mismatch if we construct carefully, 
                # but better to pad explicitly or use Series.
                # However, scalars should be broadcasted?
                # If it's a short array (not scalar), padding with NaN is appropriate.
                # If it's empty, pad all NaN.
                if len(arr) == 0:
                     df_data[k] = [np.nan] * max_len
                else:
                     # Pad end
                     padded = list(arr) + [np.nan] * (max_len - len(arr))
                     df_data[k] = padded
            else:
                df_data[k] = arr
        else:
            # Handle scalar - broadcast
            df_data[k] = [v] * max_len

    df = pd.DataFrame(df_data)
    
    # Let write failures propagate so callers cannot report a false success.
    df.to_excel(path, index=False)

def load_excel(path: Path) -> Dict[str, Any]:
    """Load Excel file to dictionary."""
    if not path.exists():
        return {}
    
    try:
        df = pd.read_excel(path)
        return df.to_dict(orient='list')
    except Exception as e:
        print(f"Error loading Excel {path}: {e}")
        return {}

def load_csv(path: Path) -> Dict[str, Any]:
    """Load CSV file to dictionary."""
    if not path.exists():
        return {}
    
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient='list')
    except Exception as e:
        print(f"Error loading CSV {path}: {e}")
        return {}

def load_parameter_file(path: Path) -> Dict[str, Any]:
    """
    Load parameter file (XLSX or CSV).
    Prioritize XLSX if both exist (using path stem).
    """
    p = Path(path)
    stem = p.stem
    parent = p.parent
    
    xlsx_path = parent / f"{stem}.xlsx"
    csv_path = parent / f"{stem}.csv"
    
    if xlsx_path.exists():
        return load_excel(xlsx_path)
    elif csv_path.exists():
        return load_csv(csv_path)
    
    return {}

def get_fast_param_path(path: Path) -> Path:
    p = Path(path)
    return p.parent / f"{p.stem}{FAST_PARAM_SUFFIX}"

def resolve_parameter_source(path: Path) -> Tuple[Optional[Path], Optional[str]]:
    p = Path(path)
    stem = p.stem
    parent = p.parent
    fast_path = parent / f"{stem}{FAST_PARAM_SUFFIX}"
    xlsx_path = parent / f"{stem}.xlsx"
    csv_path = parent / f"{stem}.csv"
    if fast_path.exists():
        return fast_path, "fastdb"
    if xlsx_path.exists():
        return xlsx_path, "xlsx"
    if csv_path.exists():
        return csv_path, "csv"
    return None, None

def ensure_fast_parameter_db(path: Path) -> Optional[Path]:
    p = Path(path)
    stem = p.stem
    parent = p.parent
    fast_path = parent / f"{stem}{FAST_PARAM_SUFFIX}"
    xlsx_path = parent / f"{stem}.xlsx"
    csv_path = parent / f"{stem}.csv"

    source_path = None
    source_kind = None
    if xlsx_path.exists():
        source_path = xlsx_path
        source_kind = "xlsx"
    elif csv_path.exists():
        source_path = csv_path
        source_kind = "csv"

    if source_path is None:
        return fast_path if _is_fastdb_valid(fast_path) else None

    needs_rebuild = (not fast_path.exists()) or (source_path.stat().st_mtime > fast_path.stat().st_mtime)
    if not needs_rebuild and not _is_fastdb_valid(fast_path):
        needs_rebuild = True
    if not needs_rebuild:
        return fast_path

    try:
        if source_kind == "xlsx":
            df = pd.read_excel(source_path)
        else:
            df = pd.read_csv(source_path)
        if "Time_s" not in df.columns:
            return None
        df["Time_s"] = pd.to_numeric(df["Time_s"], errors='coerce')
        df = df[np.isfinite(df["Time_s"])].copy()
        if len(df) == 0:
            return None
        df.sort_values("Time_s", inplace=True)
        save_fast_parameter_db(source_path, df)
        return fast_path if _is_fastdb_valid(fast_path) else None
    except Exception:
        return None

def save_fast_parameter_db(path: Path, df: pd.DataFrame) -> None:
    fast_path = get_fast_param_path(path)
    if not fast_path.parent.exists():
        fast_path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    if "Time_s" not in df_to_save.columns:
        raise ValueError("Time_s column is required for fast parameter db")
    df_to_save["Time_s"] = pd.to_numeric(df_to_save["Time_s"], errors='coerce')
    df_to_save = df_to_save[np.isfinite(df_to_save["Time_s"])].copy()
    if len(df_to_save) == 0:
        raise ValueError("No valid Time_s rows for fast parameter db")
    df_to_save.sort_values("Time_s", inplace=True)
    conn = sqlite3.connect(str(fast_path))
    try:
        df_to_save.to_sql("params", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_params_time ON params(Time_s)")
        conn.commit()
    finally:
        conn.close()

def load_fastdb_columns(path: Path) -> List[str]:
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.execute("PRAGMA table_info(params)")
        return [row[1] for row in cur.fetchall()]
    finally:
        conn.close()

def load_fastdb_window(path: Path, start_sec: float, end_sec: float, cols: List[str]) -> Optional[pd.DataFrame]:
    if "Time_s" not in cols:
        cols = ["Time_s"] + cols
    select_cols = ",".join([f'"{c}"' for c in cols])
    query = (
        f'SELECT {select_cols} FROM params '
        f'WHERE CAST("Time_s" AS REAL) >= ? AND CAST("Time_s" AS REAL) <= ? '
        f'ORDER BY CAST("Time_s" AS REAL)'
    )
    conn = sqlite3.connect(str(path))
    try:
        df = pd.read_sql_query(query, conn, params=(float(start_sec), float(end_sec)))
        if len(df) == 0:
            return None
        if "Time_s" in df.columns:
            df["Time_s"] = pd.to_numeric(df["Time_s"], errors='coerce')
            df = df[np.isfinite(df["Time_s"])]
            if len(df) == 0:
                return None
        return df
    finally:
        conn.close()
