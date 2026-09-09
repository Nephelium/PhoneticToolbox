# -*- coding: utf-8 -*-
"""嵌入式「语音标注对齐」网页服务。

在主程序进程内以守护线程运行一个仅监听 127.0.0.1 的小型 HTTP 服务，
提供网页版 Praat TextGrid 编辑器（静态页面 + REST API）。
用户点击主页按钮后即自动打开浏览器使用，无需手动启动任何服务器。

API:
    GET  /api/list                       当前语料列表
    GET  /api/item?id=                   单条语料（TextGrid 文本 + 音频 URL + lab 词表）
    GET  /api/audio?id=                  音频字节流
    GET  /api/lip?id=                    唇形曲线（audio_recording.pkl）
    GET  /api/scan?path=                 扫描新文件夹
    GET  /api/choose-folder?initial=     弹出系统文件夹选择框（由 GUI 注入回调）
    POST /api/save      {id,textgrid,suffix}  保存 TextGrid（原名+后缀，不覆盖原文件）
    POST /api/lip/save  {id,offset}           保存唇形时间偏移（写回 pkl metadata）
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import pickle
import posixpath
import tempfile
import threading
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from phonetic_toolbox.utils import get_resource_path
from phonetic_toolbox.services.io.lip import resolve_lip_time_axis

logger = logging.getLogger(__name__)

STATIC_ROOT = Path(get_resource_path(r"phonetic_toolbox\gui\resources\web_praat_editor"))


def _settings():
    from PyQt6.QtCore import QSettings

    return QSettings("PhoneticToolbox", "WebPraatEditor")


def _load_last_root() -> str:
    try:
        return str(_settings().value("lastRoot", "") or "")
    except Exception:
        return ""


def _save_last_root(root: str) -> None:
    try:
        _settings().setValue("lastRoot", root)
    except Exception:
        pass


@dataclass(frozen=True)
class Item:
    id: str
    wav: Path
    textgrid: Path
    rel: str


def decode_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def preferred_textgrid_for_wav(wav: Path) -> Path | None:
    candidates = [
        wav.with_name(f"{wav.stem}_webedit.TextGrid"),
        wav.with_name(f"{wav.stem}_webedit.textgrid"),
        wav.with_name(f"{wav.stem}_post.TextGrid"),
        wav.with_name(f"{wav.stem}_post.textgrid"),
        wav.with_name(f"{wav.stem}_auto.TextGrid"),
        wav.with_name(f"{wav.stem}_auto.textgrid"),
        wav.with_suffix(".TextGrid"),
        wav.with_suffix(".textgrid"),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def read_lab_words_for_wav(wav: Path) -> list[str]:
    lab = wav.with_suffix(".lab")
    if not lab.exists():
        return []
    return [word.strip() for word in decode_text(lab).split() if word.strip()]


def _find_lip_pkl(wav: Path) -> tuple[Path | None, Path | None]:
    """Find audio_recording.pkl and audio_recording_timestamps.pkl next to wav."""
    rec = wav.with_name("audio_recording.pkl")
    ts = wav.with_name("audio_recording_timestamps.pkl")
    return (rec if rec.exists() else None,
            ts if ts.exists() else None)


def read_lip_data_json(wav: Path) -> dict:
    """Read lip pkl files and return JSON-serializable lip data for the web."""
    rec_path, ts_path = _find_lip_pkl(wav)
    if rec_path is None:
        return {"available": False}

    try:
        with open(rec_path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    try:
        axis = resolve_lip_time_axis(data, rec_path)
    except (TypeError, ValueError) as exc:
        return {"available": False, "error": str(exc)}
    src_times = axis.times
    manual_offset = axis.manual_offset

    open_vals = data.get("open")
    if open_vals is None or len(open_vals) != axis.sample_count:
        return {"available": False, "error": "No lip openness data"}

    open_arr = np.array(open_vals, dtype=float)[axis.indices]
    valid = ~np.isnan(open_arr)
    if np.count_nonzero(valid) < 2:
        return {"available": False, "error": "Not enough valid lip openness data"}

    def _sanitize(arr: np.ndarray) -> list:
        """Replace NaN/Inf with None so JSON stays valid."""
        out = arr.tolist()
        for i, v in enumerate(out):
            if isinstance(v, float) and not np.isfinite(v):
                out[i] = None
        return out

    result = {
        "available": True,
        "times": _sanitize(src_times),
        "lipOpen": _sanitize(open_arr),
        "offset": manual_offset,
    }

    outer_width_vals = data.get("outer_width")
    if outer_width_vals is not None and len(outer_width_vals) == axis.sample_count:
        outer_arr = np.array(outer_width_vals, dtype=float)[axis.indices]
        if np.count_nonzero(~np.isnan(outer_arr)) >= 2:
            result["lipWidth"] = _sanitize(outer_arr)

    return result


def save_lip_offset(wav: Path, new_offset: float) -> tuple[bool, str]:
    """Update lip_manual_offset in audio_recording.pkl metadata.

    This is the canonical location — services/io/lip.py reads it from here.
    Returns (ok, message).
    """
    rec_path = wav.with_name("audio_recording.pkl")
    if not rec_path.exists():
        return False, f"recording pkl not found: {rec_path}"
    try:
        with open(rec_path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        return False, f"Failed to read {rec_path}: {exc}"

    if not isinstance(data.get("metadata"), dict):
        data["metadata"] = {}
    data["metadata"]["lip_manual_offset"] = new_offset

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=rec_path.parent,
            prefix=f".{rec_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            pickle.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, rec_path)
        logger.info("Lip offset %.6fs written to %s", new_offset, rec_path)
        return True, str(rec_path)
    except Exception as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to remove temporary lip pkl: %s", temp_path)
        return False, f"Failed to write {rec_path}: {exc}"


def find_items(root: Path) -> list[Item]:
    items: list[Item] = []
    # IDs belong to one scan and one absolute path. A stale browser tab must
    # never resolve its old ID to a different recording after a rescan.
    scan_id = uuid.uuid4()
    for wav in sorted(root.rglob("*.wav")):
        textgrid = preferred_textgrid_for_wav(wav)
        if textgrid is None:
            continue
        rel = str(wav.relative_to(root)).replace("\\", "/")
        item_id = uuid.uuid5(scan_id, str(wav.resolve())).hex
        items.append(Item(id=item_id, wav=wav, textgrid=textgrid, rel=rel))
    return items


def json_bytes(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


class EditorHandler(BaseHTTPRequestHandler):
    server: "EditorServer"

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003 - stdlib signature
        logger.debug("[web_praat] " + fmt, *args)

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data: object, status: int = 200) -> None:
        self.send_bytes(json_bytes(data), "application/json; charset=utf-8", status)

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status)

    def item_from_query(self, query: dict[str, list[str]]) -> Item | None:
        item_id = query.get("id", [""])[0]
        return self.server.items_by_id.get(item_id)

    def _items_payload(self, items=None) -> list[dict]:
        return [
            {
                "id": item.id,
                "rel": item.rel,
                "wav": item.wav.name,
                "textgrid": (preferred_textgrid_for_wav(item.wav) or item.textgrid).name,
            }
            for item in (self.server.items if items is None else items)
        ]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/api/list":
            root = str(self.server.root) if self.server.root else ""
            self.send_json({"root": root, "items": self._items_payload()})
            return
        if parsed.path == "/api/item":
            item = self.item_from_query(query)
            if item is None:
                self.send_error_json("语料列表已变更或文件不存在，请先保留当前修改，再重新选择语料文件夹。", HTTPStatus.NOT_FOUND)
                return
            textgrid = preferred_textgrid_for_wav(item.wav) or item.textgrid
            self.send_json(
                {
                    "id": item.id,
                    "rel": item.rel,
                    "wavName": item.wav.name,
                    "textgridName": textgrid.name,
                    "textgrid": decode_text(textgrid),
                    "labWords": read_lab_words_for_wav(item.wav),
                    "audioUrl": f"/api/audio?id={item.id}",
                }
            )
            return
        if parsed.path == "/api/audio":
            item = self.item_from_query(query)
            if item is None:
                self.send_error_json("语料列表已变更或文件不存在，请先保留当前修改，再重新选择语料文件夹。", HTTPStatus.NOT_FOUND)
                return
            self.send_bytes(item.wav.read_bytes(), "audio/wav")
            return
        if parsed.path == "/api/lip":
            item = self.item_from_query(query)
            if item is None:
                self.send_error_json("语料列表已变更或文件不存在，请先保留当前修改，再重新选择语料文件夹。", HTTPStatus.NOT_FOUND)
                return
            self.send_json(read_lip_data_json(item.wav))
            return
        if parsed.path == "/api/scan":
            path_str = query.get("path", [""])[0]
            if not path_str:
                self.send_error_json("Missing path parameter")
                return
            new_root = Path(path_str).resolve()
            if not new_root.exists():
                self.send_error_json(f"文件夹不存在：{new_root}")
                return
            if not new_root.is_dir():
                self.send_error_json(f"不是文件夹：{new_root}")
                return
            items = find_items(new_root)
            self.server.root = new_root
            self.server.items = items
            self.server.items_by_id = {item.id: item for item in items}
            _save_last_root(str(new_root))
            logger.info("Scanned %s -> %d items", new_root, len(items))
            self.send_json({"root": str(new_root), "items": self._items_payload(items)})
            return
        if parsed.path == "/api/choose-folder":
            initial_str = query.get("initial", [""])[0]
            picker = self.server.folder_picker
            if picker is None:
                self.send_error_json("文件夹选择框不可用，请手动输入路径")
                return
            try:
                selected = picker(initial_str or (str(self.server.root) if self.server.root else ""))
            except Exception as exc:
                self.send_error_json(f"文件夹选择框打开失败：{exc}")
                return
            self.send_json({"path": selected or ""})
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            self.send_error_json(f"Invalid JSON: {exc}")
            return

        if parsed.path == "/api/save":
            item = self.server.items_by_id.get(str(payload.get("id", "")))
            if item is None:
                self.send_error_json("语料列表已变更或文件不存在，请先保留当前修改，再重新选择语料文件夹。", HTTPStatus.NOT_FOUND)
                return
            textgrid = str(payload.get("textgrid", ""))
            if not textgrid.strip():
                self.send_error_json("Empty TextGrid")
                return
            suffix = str(payload.get("suffix", self.server.save_suffix))
            output = item.wav.with_name(f"{item.wav.stem}{suffix}.TextGrid")
            output.write_text(textgrid, encoding="utf-8", newline="\n")
            logger.info("Saved TextGrid -> %s", output)
            self.send_json({"ok": True, "output": str(output)})
            return

        if parsed.path == "/api/lip/save":
            item = self.server.items_by_id.get(str(payload.get("id", "")))
            if item is None:
                self.send_error_json("语料列表已变更或文件不存在，请先保留当前修改，再重新选择语料文件夹。", HTTPStatus.NOT_FOUND)
                return
            new_offset = float(payload.get("offset", 0))
            ok, msg = save_lip_offset(item.wav, new_offset)
            if not ok:
                self.send_error_json(msg)
                return
            self.send_json({"ok": True, "offset": new_offset, "path": msg})
            return

        self.send_error_json("Unknown endpoint", HTTPStatus.NOT_FOUND)

    def serve_static(self, request_path: str) -> None:
        clean_path = posixpath.normpath(unquote(request_path)).lstrip("/")
        if clean_path in ("", "."):
            clean_path = "index.html"
        target = (STATIC_ROOT / clean_path).resolve()
        try:
            target.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self.send_error_json("Invalid static path", HTTPStatus.FORBIDDEN)
            return
        if not target.exists() or not target.is_file():
            self.send_error_json("Not found", HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix.lower() in {".html", ".css", ".js", ".dict", ".lab"}:
            content_type += "; charset=utf-8"
        self.send_bytes(target.read_bytes(), content_type)


class EditorServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        root: Path | None,
        items: list[Item],
        save_suffix: str,
        folder_picker=None,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.root = root
        self.items = items
        self.items_by_id = {item.id: item for item in items}
        self.save_suffix = save_suffix
        self.folder_picker = folder_picker


_server: EditorServer | None = None
_server_thread: threading.Thread | None = None
_lock = threading.Lock()


def ensure_server(folder_picker=None, save_suffix: str = "_webedit") -> str:
    """Start the embedded editor server once and return its URL.

    Args:
        folder_picker: optional callable(initial_path: str) -> str that opens a
            native folder dialog (injected by the GUI so it runs on the Qt main
            thread). May be replaced on subsequent calls.
        save_suffix: default suffix used when the client does not send one.
    """
    global _server, _server_thread
    with _lock:
        if _server is not None:
            if folder_picker is not None:
                _server.folder_picker = folder_picker
            host, port = _server.server_address[:2]
            return f"http://127.0.0.1:{port}/"

        last_root = _load_last_root()
        root: Path | None = None
        items: list[Item] = []
        if last_root:
            candidate = Path(last_root)
            if candidate.is_dir():
                root = candidate.resolve()
                try:
                    items = find_items(root)
                except Exception:
                    logger.exception("Failed to scan last root %s", root)
                    items = []

        server = EditorServer(
            ("127.0.0.1", 0),
            EditorHandler,
            root=root,
            items=items,
            save_suffix=save_suffix,
            folder_picker=folder_picker,
        )
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
        thread.start()
        _server = server
        _server_thread = thread
        host, port = server.server_address[:2]
        url = f"http://127.0.0.1:{port}/"
        logger.info("Web Praat editor server started at %s (root=%s, items=%d)", url, root, len(items))
        return url


def shutdown_server() -> None:
    """Stop the embedded server (mainly for tests)."""
    global _server, _server_thread
    with _lock:
        if _server is not None:
            _server.shutdown()
            _server.server_close()
            _server = None
            _server_thread = None
