"""Register real processed WAV examples from a verified manual capture report."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "manual"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    options = parser.parse_args()
    report = json.loads(options.report.read_text(encoding="utf-8"))
    if report.get("success") is not True:
        raise ValueError("Capture report is unfinished")
    project_path = MANUAL / "project.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    assets = {asset["id"]: asset for asset in project["assets"]}
    registered = []

    for example in report.get("audioAssets", []):
        asset_id = example["id"]
        if not re.fullmatch(r"[a-z0-9-]+", asset_id):
            raise ValueError("Invalid audio identity")
        if example.get("kind") != "audio" or example.get("sourceType") != "v3 实际处理":
            raise ValueError("Only real processed audio may be registered")
        if example.get("distribution") != "software-only" or example.get("git") is not False:
            raise ValueError("Private audio must remain software-only")
        source = Path(example["file"])
        if source.suffix.lower() != ".wav":
            raise ValueError("This registrar accepts WAV only")
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != example["sha256"]:
            raise ValueError(f"Audio changed: {asset_id}")
        if asset_id in assets and assets[asset_id]["sha256"] != digest:
            asset_id += "-v3-" + digest[:8]
        relative = f"assets/software-only/audio/{asset_id}.wav"
        target = MANUAL / relative
        if not target.resolve().is_relative_to(MANUAL.resolve()):
            raise ValueError("Unsafe audio path")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("Audio identity collides with different bytes")
        if not target.exists():
            shutil.copyfile(source, target)
        assets[asset_id] = {
            "id": asset_id,
            "path": relative,
            "kind": "audio",
            "mime": "audio/wav",
            "sha256": digest,
            "sourceType": "v3 实际处理音频",
            "source": "作者指定录音及 V3 本地实际处理，具体方法见拍摄报告",
            "caption": example["caption"],
            "distribution": "software-only",
            "git": False,
            "sampleRate": example["sampleRate"],
            "channels": example["channels"],
            "duration": example["duration"],
            "phaseMethod": example["phaseMethod"],
            "captureReport": str(options.report.resolve().relative_to(ROOT.resolve())),
        }
        registered.append({"captureId": example["id"], "assetId": asset_id})

    project["assets"] = list(assets.values())
    project_path.write_text(json.dumps(project, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"registered": registered, "assets": len(assets)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
