from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

from phonetic_toolbox.models.mfa_models import (
    MFAAlignmentConfig,
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
        app_ok, app_message = self._ensure_runtime_app(project_dir)
        if not app_ok:
            return MFAAutoAlignmentLaunchResult(
                success=False,
                message=app_message,
                project_dir=str(project_dir),
        )

        batch_file = project_dir / "auto_alignment.bat"
        command = self._build_batch_command(batch_file)

        try:
            process = subprocess.Popen(
                command,
                cwd=str(project_dir),
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
        beam: int | None = None,
        retry_beam: int | None = None,
    ) -> MFAAlignmentRunResult:
        config = MFAAlignmentConfig()
        if beam is not None:
            config.beam = beam
        if retry_beam is not None:
            config.retry_beam = retry_beam
        if config.retry_beam <= config.beam:
            config.retry_beam = config.beam * 4

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

        if self._should_use_external_runtime():
            return self._run_alignment_external(
                audio_path=audio_path,
                dict_path=dict_path,
                acoustic_path=acoustic_path,
                output_path=output_path,
                config=config,
            )

        pipeline = MFAAlignmentPipeline()
        try:
            success, message = pipeline.run(
                audio_path=audio_path,
                dict_path=dict_path,
                acoustic_path=acoustic_path,
                output_path=output_path,
                beam=config.beam,
                retry_beam=config.retry_beam,
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

    def _run_alignment_external(
        self,
        audio_path: str,
        dict_path: str,
        acoustic_path: str,
        output_path: str,
        config: MFAAlignmentConfig,
    ) -> MFAAlignmentRunResult:
        project_dir = self._resolve_project_dir()
        if project_dir is None:
            return MFAAlignmentRunResult(
                success=False,
                message=(
                    "未找到 auto_alignment 目录。请将 auto_alignment 文件夹放在 "
                    "PhoneticToolbox.exe 同级目录。"
                ),
                output_path=output_path,
            )
        check_ok, message = self._validate_runtime(project_dir)
        if not check_ok:
            return MFAAlignmentRunResult(
                success=False,
                message=message,
                output_path=output_path,
            )
        app_ok, app_message = self._ensure_runtime_app(project_dir)
        if not app_ok:
            return MFAAlignmentRunResult(
                success=False,
                message=app_message,
                output_path=output_path,
            )

        python_exe = project_dir / "env" / "python.exe"
        runner = (
            project_dir
            / "app"
            / "phonetic_toolbox"
            / "services"
            / "pipelines"
            / "mfa_alignment_cli.py"
        )
        if not python_exe.exists():
            python_exe = project_dir / "env" / "Scripts" / "python.exe"
        if not python_exe.exists():
            return MFAAlignmentRunResult(
                success=False,
                message=f"auto_alignment 环境中找不到 python.exe: {python_exe}",
                output_path=output_path,
            )
        if not runner.exists():
            return MFAAlignmentRunResult(
                success=False,
                message=f"auto_alignment 运行入口缺失: {runner}",
                output_path=output_path,
            )

        result_file = Path(tempfile.mkdtemp(prefix="mfa_ptbx_result_")) / "result.json"
        command = [
            str(python_exe),
            str(runner),
            "--audio-path",
            audio_path,
            "--dict-path",
            dict_path,
            "--acoustic-path",
            acoustic_path,
            "--output-path",
            output_path,
            "--beam",
            str(config.beam),
            "--retry-beam",
            str(config.retry_beam),
            "--result-json",
            str(result_file),
        ]

        env = self._external_runtime_env(project_dir)
        try:
            completed = subprocess.run(
                command,
                cwd=str(project_dir / "app"),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            result = {}
            if result_file.exists():
                result = json.loads(result_file.read_text(encoding="utf-8"))
            detail_parts = []
            if completed.stdout:
                detail_parts.append(completed.stdout)
            if completed.stderr:
                detail_parts.append(completed.stderr)
            if result.get("detail"):
                detail_parts.append(result["detail"])
            if result:
                return MFAAlignmentRunResult(
                    success=bool(result.get("success")),
                    message=str(result.get("message", "")),
                    output_path=output_path,
                    detail="\n".join(detail_parts),
                )
            return MFAAlignmentRunResult(
                success=False,
                message=f"MFA 外部进程退出，代码: {completed.returncode}",
                output_path=output_path,
                detail="\n".join(detail_parts),
            )
        except Exception as exc:
            return MFAAlignmentRunResult(
                success=False,
                message=f"执行 MFA 外部环境失败: {exc}",
                output_path=output_path,
                detail=traceback.format_exc(),
            )
        finally:
            shutil.rmtree(result_file.parent, ignore_errors=True)

    def _resolve_project_dir(self) -> Optional[Path]:
        candidate_dirs: list[Path] = []

        if self.project_dir:
            candidate_dirs.append(Path(self.project_dir))

        env_dir = os.getenv("PHONETIC_TOOLBOX_ALIGNMENT_PROJECT_DIR")
        if env_dir:
            candidate_dirs.append(Path(env_dir))

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

    @staticmethod
    def _build_batch_command(batch_file: Path) -> list[str]:
        if sys.platform == "win32":
            return ["cmd.exe", "/c", str(batch_file)]
        return [str(batch_file)]

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

    def _ensure_runtime_app(self, project_dir: Path) -> tuple[bool, str]:
        app_dir = project_dir / "app"
        app_dir.mkdir(parents=True, exist_ok=True)

        package_target = app_dir / "phonetic_toolbox"
        for source in self._runtime_app_source_dirs("phonetic_toolbox"):
            if not source.exists() or not source.is_dir():
                continue
            if source.resolve() == package_target.resolve():
                break
            shutil.copytree(
                source,
                package_target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    ".pytest_cache",
                    ".mypy_cache",
                ),
            )
            break

        if not package_target.exists():
            return (
                False,
                "无法准备 auto_alignment/app/phonetic_toolbox 运行代码。",
            )

        for resource_name in ["Phonetic_Export", "PhoneticToolbox.ico"]:
            target = app_dir / resource_name
            for source in self._runtime_app_source_dirs(resource_name):
                if not source.exists():
                    continue
                if source.is_dir():
                    shutil.copytree(source, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, target)
                break

        return True, ""

    def _runtime_app_source_dirs(self, name: str) -> list[Path]:
        candidates: list[Path] = []
        meipass_dir = getattr(sys, "_MEIPASS", None)
        if meipass_dir:
            candidates.append(Path(meipass_dir) / name)
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / name)
        for base_dir in self._runtime_base_dirs():
            candidates.append(base_dir / name)
        return candidates

    @staticmethod
    def _should_use_external_runtime() -> bool:
        if getattr(sys, "frozen", False):
            return True
        try:
            import montreal_forced_aligner  # noqa: F401
        except Exception:
            return True
        return False

    @staticmethod
    def _external_runtime_env(project_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        env_dir = project_dir / "env"
        path_parts = [
            str(env_dir),
            str(env_dir / "Library" / "bin"),
            str(env_dir / "Scripts"),
            str(env_dir / "DLLs"),
            env.get("PATH", ""),
        ]
        env["PATH"] = os.pathsep.join(part for part in path_parts if part)
        app_dir = project_dir / "app"
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(app_dir)
            if not existing_pythonpath
            else str(app_dir) + os.pathsep + existing_pythonpath
        )
        return env

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
