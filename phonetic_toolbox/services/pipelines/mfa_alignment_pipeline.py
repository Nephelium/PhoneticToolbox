from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from phonetic_toolbox.core.transcription import decode_fs_name, encode_fs_name


class MFAAlignmentPipeline:
    def run(
        self,
        audio_path: str,
        dict_path: str,
        acoustic_path: str,
        output_path: str,
        beam: int = 10,
        retry_beam: int = 40,
    ) -> tuple[bool, str]:
        workspace_dir = tempfile.mkdtemp(prefix="mfa_ptbx_")
        encoded_corpus_dir = os.path.join(workspace_dir, "corpus")
        encoded_output_dir = os.path.join(workspace_dir, "output")
        mfa_temp_dir = os.path.join(workspace_dir, "mfa_temp")

        try:
            self._copy_and_encode_corpus(audio_path, encoded_corpus_dir)

            dictionary_suffix = Path(dict_path).suffix
            acoustic_suffix = Path(acoustic_path).suffix
            safe_dict_path = os.path.join(
                workspace_dir,
                f"dictionary{dictionary_suffix}",
            )
            safe_acoustic_path = os.path.join(
                workspace_dir,
                f"acoustic_model{acoustic_suffix}",
            )
            shutil.copy2(dict_path, safe_dict_path)
            shutil.copy2(acoustic_path, safe_acoustic_path)

            os.environ["BLAS_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            os.environ["NUMBA_DISABLE_JIT"] = "1"
            os.environ["MFA_NUM_JOBS"] = "1"
            os.environ["CALC_JOBS"] = "1"
            os.environ["NUM_JOBS"] = "1"
            os.environ["JOBLIB_MULTIPROCESSING"] = "0"

            from montreal_forced_aligner.alignment import PretrainedAligner

            aligner = PretrainedAligner(
                corpus_directory=encoded_corpus_dir,
                dictionary_path=safe_dict_path,
                acoustic_model_path=safe_acoustic_path,
                output_directory=encoded_output_dir,
                temporary_directory=mfa_temp_dir,
                clean=True,
                verbose=True,
                num_jobs=1,
                use_mp=False,
                beam=beam,
                retry_beam=retry_beam,
            )
            aligner.align()
            aligner.export_files(encoded_output_dir)
            self._decode_and_copy_output(encoded_output_dir, output_path)
            return True, f"对齐完成。\n输出路径: {output_path}"
        except Exception as exc:
            return False, f"执行出错: {exc}"
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)

    def _copy_and_encode_corpus(
        self,
        source_dir: str,
        encoded_dir: str,
    ) -> None:
        for root, _, files in os.walk(source_dir):
            rel = os.path.relpath(root, source_dir)
            encoded_parts = self._encode_relative_parts(rel)
            encoded_root = os.path.join(encoded_dir, *encoded_parts)
            os.makedirs(encoded_root, exist_ok=True)
            for file_name in files:
                src = os.path.join(root, file_name)
                encoded_name = encode_fs_name(file_name)
                dst = os.path.join(encoded_root, encoded_name)
                shutil.copy2(src, dst)

    def _decode_and_copy_output(
        self,
        encoded_output_dir: str,
        target_output_dir: str,
    ) -> None:
        os.makedirs(target_output_dir, exist_ok=True)
        for root, _, files in os.walk(encoded_output_dir):
            rel = os.path.relpath(root, encoded_output_dir)
            decoded_parts = []
            if rel != ".":
                decoded_parts = [
                    decode_fs_name(part) for part in rel.split(os.sep)
                ]
            decoded_root = os.path.join(target_output_dir, *decoded_parts)
            os.makedirs(decoded_root, exist_ok=True)
            for file_name in files:
                src = os.path.join(root, file_name)
                decoded_name = decode_fs_name(file_name)
                dst = os.path.join(decoded_root, decoded_name)
                shutil.copy2(src, dst)

    @staticmethod
    def _encode_relative_parts(relative_path: str) -> list[str]:
        if relative_path == ".":
            return []
        return [encode_fs_name(part) for part in relative_path.split(os.sep)]
