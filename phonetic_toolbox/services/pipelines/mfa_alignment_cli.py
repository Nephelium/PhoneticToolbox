from __future__ import annotations

import argparse
import json
import traceback

from phonetic_toolbox.services.pipelines.mfa_alignment_pipeline import (
    MFAAlignmentPipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--dict-path", required=True)
    parser.add_argument("--acoustic-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--beam", type=int, required=True)
    parser.add_argument("--retry-beam", type=int, required=True)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()

    try:
        success, message = MFAAlignmentPipeline().run(
            audio_path=args.audio_path,
            dict_path=args.dict_path,
            acoustic_path=args.acoustic_path,
            output_path=args.output_path,
            beam=args.beam,
            retry_beam=args.retry_beam,
        )
        result = {
            "success": success,
            "message": message,
            "detail": "",
        }
    except Exception as exc:
        result = {
            "success": False,
            "message": f"执行出错: {exc}",
            "detail": traceback.format_exc(),
        }

    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
