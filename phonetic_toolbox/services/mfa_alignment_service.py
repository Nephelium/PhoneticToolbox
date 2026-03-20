from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

from phonetic_toolbox.models.mfa_models import (
    MFAAlignmentRunResult,
    MFAAutoAlignmentLaunchResult,
)
from phonetic_toolbox.services.pipelines.mfa_alignment_pipeline import (
    MFAAlignmentPipeline,
)


class MFAAutoAlignmentService:
    def __init__(self, project_dir: Optional[str] = None):
        self.project_dir = project_dir

    def launch(self) -> MFAAutoAlignmentLaunchResult:
        project_dir = self._resolve_project_dir()
        if project_dir is None:
            return MFAAutoAlignmentLaunchResult(
                success=False,
                message=(
                    "未找到 auto_alignment 目录。请将 auto_alignment.zip 放入 "
                    "PhoneticToolbox.exe 同目录并解压。"
                ),
            )

        check_ok, message = self._validate_runtime(project_dir)
        if not check_ok:
            return MFAAutoAlignmentLaunchResult(
                success=False,
                message=message,
                project_dir=str(project_dir),
            )

        batch_file = project_dir / "auto_alignment.bat"
        command = [str(batch_file)]

        try:
            process = subprocess.Popen(
                command,
                cwd=str(project_dir),
                shell=True,
            )
            return MFAAutoAlignmentLaunchResult(
                success=True,
                message="已启动 MFA 自动标注程序。",
                project_dir=str(project_dir),
                batch_file=str(batch_file),
                working_directory=str(project_dir),
                command=command,
                process_id=process.pid,
            )
        except Exception as exc:
            return MFAAutoAlignmentLaunchResult(
                success=False,
                message=f"启动 MFA 自动标注失败: {exc}",
                project_dir=str(project_dir),
                batch_file=str(batch_file),
                working_directory=str(project_dir),
                command=command,
            )

    def run_alignment(
        self,
        audio_path: str,
        dict_path: str,
        acoustic_path: str,
        output_path: str,
    ) -> MFAAlignmentRunResult:
        if not os.path.isdir(audio_path):
            return MFAAlignmentRunResult(
                success=False,
                message="音频路径不是有效文件夹。",
                output_path=output_path,
            )
        if not os.path.isfile(dict_path):
            return MFAAlignmentRunResult(
                success=False,
                message="字典路径不是有效文件。",
                output_path=output_path,
            )
        if not os.path.isfile(acoustic_path):
            return MFAAlignmentRunResult(
                success=False,
                message="声学模型路径不是有效文件。",
                output_path=output_path,
            )
        if not output_path:
            return MFAAlignmentRunResult(
                success=False,
                message="输出路径不能为空。",
                output_path=output_path,
            )

        pipeline = MFAAlignmentPipeline()
        try:
            success, message = pipeline.run(
                audio_path=audio_path,
                dict_path=dict_path,
                acoustic_path=acoustic_path,
                output_path=output_path,
            )
            return MFAAlignmentRunResult(
                success=success,
                message=message,
                output_path=output_path,
            )
        except Exception as exc:
            return MFAAlignmentRunResult(
                success=False,
                message=f"执行出错: {exc}",
                output_path=output_path,
                detail=traceback.format_exc(),
            )

    def _resolve_project_dir(self) -> Optional[Path]:
        candidate_dirs: list[Path] = []

        if self.project_dir:
            candidate_dirs.append(Path(self.project_dir))

        env_dir = os.getenv("PHONETIC_TOOLBOX_ALIGNMENT_PROJECT_DIR")
        if env_dir:
            candidate_dirs.append(Path(env_dir))

        candidate_dirs.append(
            Path(r"D:\project\【中山大学】\PhoneticToolbox\auto_alignment")
        )

        for base_dir in self._runtime_base_dirs():
            candidate_dirs.append(base_dir / "auto_alignment")
            candidate_dirs.append(base_dir)

        checked: set[str] = set()
        for candidate in candidate_dirs:
            key = str(candidate)
            if key in checked:
                continue
            checked.add(key)
            if candidate.exists() and candidate.is_dir():
                if (candidate / "auto_alignment.bat").exists():
                    return candidate
        return None

    def _validate_runtime(self, project_dir: Path) -> tuple[bool, str]:
        required_paths = [
            project_dir / "auto_alignment.bat",
            project_dir / "env" / "Scripts" / "activate.bat",
        ]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            missing_display = "\n".join(str(path) for path in missing_paths)
            return (
                False,
                (
                    "auto_alignment 环境不完整，缺少以下文件：\n"
                    f"{missing_display}\n\n"
                    "请将 auto_alignment.zip 放入 PhoneticToolbox.exe 同目录并解压。"
                ),
            )
        return True, ""

    @staticmethod
    def _runtime_base_dirs() -> list[Path]:
        candidate_dirs: list[Path] = []
        if getattr(sys, "frozen", False):
            candidate_dirs.append(Path(sys.executable).resolve().parent)
            meipass_dir = getattr(sys, "_MEIPASS", None)
            if meipass_dir:
                candidate_dirs.append(Path(meipass_dir))

        candidate_dirs.append(Path.cwd())
        candidate_dirs.append(Path(sys.executable).resolve().parent)
        candidate_dirs.append(Path(sys.argv[0]).resolve().parent)
        candidate_dirs.append(Path(__file__).resolve().parents[2])

        unique_dirs: list[Path] = []
        seen: set[str] = set()
        for path_obj in candidate_dirs:
            key = str(path_obj)
            if key in seen:
                continue
            seen.add(key)
            unique_dirs.append(path_obj)
        return unique_dirs
