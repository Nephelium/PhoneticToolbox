from __future__ import annotations

import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

from phonetic_toolbox.models.perception_models import PerceptionLaunchResult


class PerceptionExperimentService:
    def __init__(self, project_dir: Optional[str] = None):
        self.project_dir = project_dir
        self._debug_log_path: Optional[str] = None

    def launch(self) -> PerceptionLaunchResult:
        project_dir = self._resolve_project_dir()
        if project_dir is None:
            msg = "未找到 perception_experiment 项目目录，请检查目录配置。"
            if self._debug_log_path:
                msg += f"\n\n诊断日志已保存至：\n{self._debug_log_path}\n请将该文件发送给开发者以便排查。"
            return PerceptionLaunchResult(
                success=False,
                message=msg,
            )

        html_path = self._resolve_html_path(project_dir)
        if html_path is None:
            return PerceptionLaunchResult(
                success=False,
                message="未找到可打开的感知实验 HTML 页面。",
                working_directory=str(project_dir),
            )

        try:
            opened = self._open_html(html_path)
            if not opened:
                return PerceptionLaunchResult(
                    success=False,
                    message="系统未能打开感知实验页面，请检查默认浏览器关联设置。",
                    html_path=str(html_path),
                    working_directory=str(project_dir),
                )
            return PerceptionLaunchResult(
                success=True,
                message="已打开感知实验页面。",
                html_path=str(html_path),
                working_directory=str(project_dir),
            )
        except Exception as exc:
            return PerceptionLaunchResult(
                success=False,
                message=f"打开感知实验页面失败: {exc}",
                html_path=str(html_path),
                working_directory=str(project_dir),
            )

    def _resolve_project_dir(self) -> Optional[Path]:
        candidate_dirs: list[Path] = []

        if self.project_dir:
            candidate_dirs.append(Path(self.project_dir))

        env_dir = os.getenv("PHONETIC_TOOLBOX_PERCEPTION_PROJECT_DIR")
        if env_dir:
            candidate_dirs.append(Path(env_dir))

        runtime_base_dirs = self._runtime_base_dirs()
        for base_dir in runtime_base_dirs:
            candidate_dirs.append(
                base_dir
                / "phonetic_toolbox"
                / "gui"
                / "resources"
                / "perception_experiment"
            )
            candidate_dirs.append(
                base_dir / "gui" / "resources" / "perception_experiment"
            )

        candidate_dirs.append(
            self._repo_root()
            / "phonetic_toolbox"
            / "gui"
            / "resources"
            / "perception_experiment"
        )
        for base_dir in runtime_base_dirs:
            candidate_dirs.append(base_dir / "perception_experiment")
        candidate_dirs.append(self._repo_root() / "perception_experiment")

        checked: set[str] = set()
        for candidate in candidate_dirs:
            key = (
                str(candidate.resolve())
                if candidate.exists()
                else str(candidate)
            )
            if key in checked:
                continue
            checked.add(key)
            if candidate.exists() and candidate.is_dir():
                return candidate
        discovered = self._discover_project_dir()
        if discovered is not None:
            return discovered

        self._write_debug_log(candidate_dirs, runtime_base_dirs)
        return None

    def _write_debug_log(
        self,
        candidate_dirs: list[Path],
        runtime_base_dirs: list[Path],
    ) -> None:
        desktop = Path.home() / "Desktop"
        log_path = desktop / "perception_debug.log"
        try:
            lines: list[str] = []
            lines.append(f"=== 感知实验路径解析诊断日志 ===")
            lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"sys.frozen: {getattr(sys, 'frozen', None)}")
            lines.append(f"sys._MEIPASS: {getattr(sys, '_MEIPASS', None)}")
            lines.append(f"sys.executable: {sys.executable}")
            lines.append(f"sys.argv[0]: {sys.argv[0] if sys.argv else 'N/A'}")
            lines.append(f"cwd: {Path.cwd()}")
            lines.append(f"project_dir (显式传入): {self.project_dir}")
            lines.append(f"env PHONETIC_TOOLBOX_PERCEPTION_PROJECT_DIR: {os.getenv('PHONETIC_TOOLBOX_PERCEPTION_PROJECT_DIR')}")
            lines.append(f"")
            lines.append(f"--- _runtime_base_dirs() 返回值 ---")
            for i, d in enumerate(runtime_base_dirs):
                lines.append(f"  [{i}] {d}  (exists={d.exists()})")
            lines.append(f"")
            lines.append(f"--- 候选路径检查结果 ---")
            checked: set[str] = set()
            for candidate in candidate_dirs:
                key = (
                    str(candidate.resolve())
                    if candidate.exists()
                    else str(candidate)
                )
                if key in checked:
                    continue
                checked.add(key)
                status = "EXISTS(dir)" if (candidate.exists() and candidate.is_dir()) else ("EXISTS(file)" if candidate.exists() else "MISSING")
                lines.append(f"  {status} | {candidate}")
            lines.append(f"")
            lines.append(f"--- _discover_project_dir() 递归搜索 ---")
            for root in self._common_search_roots():
                if not root.exists():
                    lines.append(f"  SKIP (root不存在): {root}")
                    continue
                lines.append(f"  搜索 root: {root}")
                found = self._find_named_dir(root, "perception_experiment", 4)
                if found is not None:
                    lines.append(f"    找到: {found}")
                else:
                    lines.append(f"    未找到")

            content = "\n".join(lines)
            log_path.write_text(content, encoding="utf-8")
            self._debug_log_path = str(log_path)
        except Exception:
            self._debug_log_path = None

    @staticmethod
    def _resolve_html_path(project_dir: Path) -> Optional[Path]:
        default_html = project_dir / "perception_experiment.html"
        if default_html.exists():
            return default_html

        html_files = sorted(project_dir.glob("*.html"))
        if html_files:
            return html_files[0]
        return None

    @staticmethod
    def _open_html(html_path: Path) -> bool:
        if hasattr(os, "startfile"):
            try:
                os.startfile(str(html_path))
                return True
            except OSError:
                pass

        html_uri = html_path.resolve().as_uri()
        return bool(webbrowser.open_new_tab(html_uri))

    @staticmethod
    def _runtime_base_dirs() -> list[Path]:
        candidate_dirs: list[Path] = []

        if getattr(sys, "frozen", False):
            candidate_dirs.append(Path(sys.executable).resolve().parent)
            meipass_dir = getattr(sys, "_MEIPASS", None)
            if meipass_dir:
                candidate_dirs.append(Path(meipass_dir))
        else:
            candidate_dirs.append(Path(__file__).resolve().parents[2])

        candidate_dirs.append(Path.cwd())
        candidate_dirs.append(Path(sys.executable).resolve().parent)
        candidate_dirs.append(Path(sys.argv[0]).resolve().parent)

        unique_dirs: list[Path] = []
        seen: set[str] = set()
        for path_obj in candidate_dirs:
            key = str(path_obj)
            if key in seen:
                continue
            seen.add(key)
            unique_dirs.append(path_obj)
        return unique_dirs

    def _discover_project_dir(self) -> Optional[Path]:
        for root in self._common_search_roots():
            if not root.exists() or not root.is_dir():
                continue
            direct = root / "perception_experiment"
            if direct.exists() and direct.is_dir():
                return direct
            nested = self._find_named_dir(
                root=root,
                target_name="perception_experiment",
                max_depth=4,
            )
            if nested is not None:
                return nested
        return None

    def _common_search_roots(self) -> list[Path]:
        roots: list[Path] = []
        for base_dir in self._runtime_base_dirs():
            roots.append(base_dir)
            if base_dir.parent != base_dir:
                roots.append(base_dir.parent)

        for base_dir in self._runtime_base_dirs():
            drive = Path(base_dir.drive + "\\") if base_dir.drive else None
            if drive is not None:
                roots.append(drive / "project")
                roots.append(drive / "projects")

        home_dir = Path.home()
        roots.append(home_dir / "project")
        roots.append(home_dir / "projects")

        unique_roots: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            unique_roots.append(root)
        return unique_roots

    @staticmethod
    def _find_named_dir(
        root: Path,
        target_name: str,
        max_depth: int,
    ) -> Optional[Path]:
        root_depth = len(root.parts)
        for current_root, dir_names, _ in os.walk(root):
            current_path = Path(current_root)
            current_depth = len(current_path.parts) - root_depth
            if current_depth > max_depth:
                dir_names[:] = []
                continue
            if target_name in dir_names:
                found = current_path / target_name
                if found.exists() and found.is_dir():
                    return found
        return None

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]
