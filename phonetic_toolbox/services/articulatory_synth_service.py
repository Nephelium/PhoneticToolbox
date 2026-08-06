from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path
from typing import Optional

from phonetic_toolbox.models.articulatory_models import (
    ArticulatorySynthLaunchResult,
)


class ArticulatorySynthService:
    def __init__(self, project_dir: Optional[str] = None):
        self.project_dir = project_dir

    def launch(self) -> ArticulatorySynthLaunchResult:
        project_dir = self._resolve_project_dir()
        if project_dir is None:
            return ArticulatorySynthLaunchResult(
                success=False,
                message="未找到发音物理模拟器资源目录，请检查目录配置。",
            )

        html_path = self._resolve_html_path(project_dir)
        if html_path is None:
            return ArticulatorySynthLaunchResult(
                success=False,
                message="未找到可打开的发音物理模拟器 HTML 页面。",
                working_directory=str(project_dir),
            )

        try:
            opened = self._open_html(html_path)
            if not opened:
                return ArticulatorySynthLaunchResult(
                    success=False,
                    message="系统未能打开发音物理模拟器页面，请检查默认浏览器关联设置。",
                    html_path=str(html_path),
                    working_directory=str(project_dir),
                )
        except Exception as exc:
            return ArticulatorySynthLaunchResult(
                success=False,
                message=f"打开发音物理模拟器页面失败: {exc}",
                html_path=str(html_path),
                working_directory=str(project_dir),
            )

        return ArticulatorySynthLaunchResult(
            success=True,
            message="已打开发音物理模拟器页面。",
            html_path=str(html_path),
            working_directory=str(project_dir),
        )

    def _resolve_project_dir(self) -> Optional[Path]:
        candidate_dirs: list[Path] = []

        if self.project_dir:
            candidate_dirs.append(Path(self.project_dir))

        env_dir = os.getenv("PHONETIC_TOOLBOX_ARTICULATORY_SYNTH_DIR")
        if env_dir:
            candidate_dirs.append(Path(env_dir))

        for base_dir in self._runtime_base_dirs():
            candidate_dirs.append(
                base_dir
                / "phonetic_toolbox"
                / "gui"
                / "resources"
                / "articulatory_synth"
            )
            candidate_dirs.append(base_dir / "articulatory_synth")

        candidate_dirs.append(
            self._repo_root()
            / "phonetic_toolbox"
            / "gui"
            / "resources"
            / "articulatory_synth"
        )

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
        return None

    @staticmethod
    def _resolve_html_path(project_dir: Path) -> Optional[Path]:
        html_path = project_dir / "articulatory_synth.html"
        if html_path.exists():
            return html_path
        return None

    @staticmethod
    def _open_html(html_path: Path) -> bool:
        if hasattr(os, "startfile"):
            try:
                os.startfile(str(html_path))
                return True
            except OSError:
                pass

        return bool(webbrowser.open_new_tab(html_path.resolve().as_uri()))

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

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]
