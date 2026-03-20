from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

from phonetic_toolbox.models.ipa_models import IPATransLaunchResult


class IPATransService:
    def __init__(self, project_dir: Optional[str] = None):
        self.project_dir = project_dir

    def launch(self) -> IPATransLaunchResult:
        resource_dir = self._resolve_resource_dir()
        if resource_dir is None:
            return IPATransLaunchResult(
                success=False,
                message="未找到 IPA 转换器资源目录，请检查目录配置。",
            )

        script_path = resource_dir / "generate_ipa_website.py"
        html_path = resource_dir / "ipa_converter.html"
        command: list[str] = []
        generator_output = ""
        generated = False
        if (
            script_path.exists()
            and self._should_generate(script_path, html_path)
        ):
            command = [self._resolve_python_executable(), str(script_path)]
            process = subprocess.run(
                command,
                cwd=str(resource_dir),
                capture_output=True,
            )
            stdout_text = self._decode_output(process.stdout)
            stderr_text = self._decode_output(process.stderr)
            output_parts = [stdout_text.strip(), stderr_text.strip()]
            generator_output = "\n".join(
                part for part in output_parts if part
            )
            if process.returncode != 0:
                return IPATransLaunchResult(
                    success=False,
                    message="IPA 网页生成失败。",
                    script_path=str(script_path),
                    html_path=str(html_path),
                    working_directory=str(resource_dir),
                    command=command,
                    generator_output=generator_output,
                )
            generated = True

        if not html_path.exists():
            if not script_path.exists():
                return IPATransLaunchResult(
                    success=False,
                    message="未找到 IPA 页面文件，也未找到生成脚本。",
                    script_path=str(script_path),
                    html_path=str(html_path),
                    working_directory=str(resource_dir),
                )
            return IPATransLaunchResult(
                success=False,
                message="IPA 网页生成完成，但未找到输出 HTML 文件。",
                script_path=str(script_path),
                html_path=str(html_path),
                working_directory=str(resource_dir),
                command=command,
                generator_output=generator_output,
            )

        try:
            opened = self._open_html(html_path)
            if not opened:
                return IPATransLaunchResult(
                    success=False,
                    message="系统未能打开 IPA 页面，请检查默认浏览器关联设置。",
                    script_path=str(script_path),
                    html_path=str(html_path),
                    working_directory=str(resource_dir),
                    command=command,
                    generator_output=generator_output,
                )
        except Exception as exc:
            return IPATransLaunchResult(
                success=False,
                message=f"打开 IPA 页面失败: {exc}",
                script_path=str(script_path),
                html_path=str(html_path),
                working_directory=str(resource_dir),
                command=command,
                generator_output=generator_output,
            )

        message = (
            "已重新生成并打开普通话转 IPA 页面。"
            if generated
            else "已直接打开普通话转 IPA 页面。"
        )
        return IPATransLaunchResult(
            success=True,
            message=message,
            script_path=str(script_path),
            html_path=str(html_path),
            working_directory=str(resource_dir),
            command=command,
            generator_output=generator_output,
        )

    def _resolve_resource_dir(self) -> Optional[Path]:
        candidate_dirs: list[Path] = []

        if self.project_dir:
            candidate_dirs.append(Path(self.project_dir))

        env_dir = os.getenv("PHONETIC_TOOLBOX_IPA_PROJECT_DIR")
        if env_dir:
            candidate_dirs.append(Path(env_dir))

        for base_dir in self._runtime_base_dirs():
            candidate_dirs.append(
                base_dir
                / "phonetic_toolbox"
                / "gui"
                / "resources"
                / "ipa_trans"
            )
            candidate_dirs.append(base_dir / "ipa_trans")

        candidate_dirs.append(
            self._repo_root()
            / "phonetic_toolbox"
            / "gui"
            / "resources"
            / "ipa_trans"
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
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _should_generate(script_path: Path, html_path: Path) -> bool:
        if not html_path.exists():
            return True
        try:
            return script_path.stat().st_mtime > html_path.stat().st_mtime
        except OSError:
            return True

    @staticmethod
    def _resolve_python_executable() -> str:
        configured_python = os.getenv("PHONETIC_TOOLBOX_IPA_PYTHON")
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
    def _decode_output(output: bytes | str | None) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        for encoding in ("utf-8", "gbk"):
            try:
                return output.decode(encoding)
            except UnicodeDecodeError:
                continue
        return output.decode("utf-8", errors="ignore")
