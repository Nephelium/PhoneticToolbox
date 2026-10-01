"""Read-only inventory of the user-authorized natural audio directory."""
from pathlib import Path
import hashlib
import json
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\13680\Desktop\project\音频数据")
OUT = ROOT / "output/validation/p17"


def main():
    rows = []
    for path in sorted(SOURCE.rglob("*.wav")):
        try:
            info = sf.info(path)
            rows.append(dict(path=str(path), relative=str(path.relative_to(SOURCE)),
                             bytes=path.stat().st_size, sample_rate=info.samplerate,
                             channels=info.channels, frames=info.frames,
                             duration=info.duration, subtype=info.subtype))
        except Exception as exc:
            rows.append(dict(path=str(path), error=str(exc)))
    valid = [r for r in rows if "error" not in r]
    ordinary = [r for r in valid if "EGG" not in r["relative"]]
    selected = {}
    for name, candidates in (
        ("short", [r for r in ordinary if .3 <= r["duration"] <= 5]),
        ("medium", [r for r in ordinary if 10 <= r["duration"] <= 60]),
        ("long", [r for r in ordinary if 60 < r["duration"] <= 120]),
        ("very_long", [r for r in ordinary if r["duration"] > 120]),
        ("egg", [r for r in valid if "EGG" in r["relative"] and r["channels"] == 2]),
    ):
        if candidates:
            # Prefer unmanipulated names, then a modest representative duration.
            candidates.sort(key=lambda r: ("Hz_" in r["relative"], r["duration"]))
            row = dict(candidates[0])
            row["sha256"] = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
            selected[name] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "natural-inventory.json").write_text(json.dumps(dict(root=str(SOURCE),
        count=len(rows), selected=selected, files=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(count=len(rows), invalid=len(rows)-len(valid), selected=selected), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
