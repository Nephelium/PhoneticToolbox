from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import logging

log = logging.getLogger(__name__)

try:
    from scipy.io import savemat, loadmat
except Exception:  # pragma: no cover
    savemat = None
    loadmat = None


def save_mat(path: Path, data: Dict[str, Any]) -> None:
    """保存为 MATLAB .mat 文件；若缺少 scipy，降级为 .npz。"""
    if savemat is not None:
        savemat(str(path), data)
        return
    # 降级方案
    log.warning("scipy 不可用，改用 .npz 保存: %s", path)
    import numpy as np
    np.savez(str(path.with_suffix(".npz")), **data)


def load_mat_any(path: Path) -> Dict[str, Any]:
    if path.suffix == ".mat" and loadmat is not None and path.exists():
        d = loadmat(str(path), squeeze_me=True)
        return {k: v for k, v in d.items() if not k.startswith("__")}
    npz = path.with_suffix(".npz")
    if npz.exists():
        import numpy as np
        data = dict(np.load(str(npz)))
        return data
    return {}