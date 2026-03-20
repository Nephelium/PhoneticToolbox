from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from phonetic_toolbox.models.lip_models import LipLaunchResult


class LipExtractionService:
    def __init__(self, project_dir: Optional[str] = None):
        self.project_dir = project_dir

    def launch(self, load_existing: bool = False) -> LipLaunchResult:
        target, command, working_directory = self._resolve_launch_context(
            load_existing=load_existing
        )
        if command is None:
            return LipLaunchResult(
                success=False,
                message="未找到 lip_feature_analysis 项目入口，请检查目录配置。",
            )

        try:
            process = subprocess.Popen(command, cwd=working_directory)
            return LipLaunchResult(
                success=True,
                message="已启动唇形提取程序。",
                target_path=str(target),
                working_directory=working_directory,
                command=command,
                process_id=process.pid,
            )
        except Exception as exc:
            return LipLaunchResult(
                success=False,
                message=f"启动唇形提取程序失败: {exc}",
                target_path=str(target),
                working_directory=working_directory,
                command=command,
            )

    def _resolve_launch_context(
        self, load_existing: bool
    ) -> tuple[Optional[Path], Optional[list[str]], str]:
        if self._is_frozen():
            command = self._build_frozen_command(load_existing=load_existing)
            working_directory = str(Path(sys.executable).resolve().parent)
            return Path(sys.executable), command, working_directory

        target = self._resolve_target()
        if target is None:
            return None, None, ""

        command = self._build_command(target, load_existing=load_existing)
        working_directory = str(target.parent)
        return target, command, working_directory

    def _resolve_target(self) -> Optional[Path]:
        for candidate in self._candidate_script_paths():
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _candidate_script_paths(self) -> list[Path]:
        candidates: list[Path] = [
            self._repo_root()
            / "phonetic_toolbox"
            / "gui"
            / "dialogs"
            / "lip_feature_analysis_standalone.py"
        ]

        for candidate_dir in self._candidate_project_dirs():
            candidates.append(candidate_dir / "lip_feature_analysis.py")
        return candidates

    def _candidate_project_dirs(self) -> list[Path]:
        candidate_dirs: list[Path] = []
        if self.project_dir:
            candidate_dirs.append(Path(self.project_dir))

        env_dir = os.getenv("PHONETIC_TOOLBOX_LIP_PROJECT_DIR")
        if env_dir:
            candidate_dirs.append(Path(env_dir))

        candidate_dirs.append(self._repo_root() / "lip_feature_analysis")
        return candidate_dirs

    def _build_command(self, target: Path, load_existing: bool) -> list[str]:
        command = [self._resolve_python_executable(), str(target)]

        if load_existing:
            command.append("--load")
        return command

    @staticmethod
    def _build_frozen_command(load_existing: bool) -> list[str]:
        command = [sys.executable, "--lip-standalone"]
        if load_existing:
            command.append("--load")
        return command

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _resolve_python_executable() -> str:
        configured_python = os.getenv("PHONETIC_TOOLBOX_LIP_PYTHON")
        if configured_python:
            return configured_python

        home_dir = Path.home()
        conda_candidates = [
            home_dir / "Miniconda3" / "envs" / "phonetic_311" / "python.exe",
            home_dir / "miniconda3" / "envs" / "phonetic_311" / "python.exe",
            home_dir / "Anaconda3" / "envs" / "phonetic_311" / "python.exe",
            home_dir / "anaconda3" / "envs" / "phonetic_311" / "python.exe",
        ]
        for candidate in conda_candidates:
            if candidate.exists():
                return str(candidate)

        return sys.executable

    @staticmethod
    def _is_frozen() -> bool:
        return bool(getattr(sys, "frozen", False))
