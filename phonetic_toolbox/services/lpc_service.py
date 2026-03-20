from pathlib import Path
import re

import matplotlib.figure
import numpy as np

from phonetic_toolbox.core.acoustic import compute_lpc_spectrum
from phonetic_toolbox.models.lpc_models import LPCSpectrumConfig, LPCSpectrumResult
from phonetic_toolbox.services.io.textgrid import TextGrid, parse_textgrid
from phonetic_toolbox.services.io.wav import read_wav_float_mono


class LPCSpectrumService:
    def load_audio(self, wav_path: Path) -> tuple[int, np.ndarray]:
        return read_wav_float_mono(wav_path)

    def read_sibling_textgrid(self, wav_path: Path) -> TextGrid | None:
        textgrid_path = wav_path.with_suffix(".TextGrid")
        if not textgrid_path.exists():
            return None
        return parse_textgrid(textgrid_path)

    def compute_spectrum(
        self,
        audio_segment: np.ndarray,
        fs: int,
        config: LPCSpectrumConfig,
    ) -> LPCSpectrumResult:
        freq_hz, magnitude_db = compute_lpc_spectrum(
            audio=audio_segment,
            fs=fs,
            order=config.order,
        )
        amp_min_db = config.amp_min_db
        amp_max_db = config.amp_max_db
        if config.dynamic_y:
            visible = magnitude_db[freq_hz <= config.freq_max_hz]
            if visible.size > 0:
                amp_min_db = float(np.min(visible) - 5.0)
                amp_max_db = float(np.max(visible) + 5.0)
        return LPCSpectrumResult(
            frequencies_hz=freq_hz,
            magnitude_db=magnitude_db,
            amp_min_db=amp_min_db,
            amp_max_db=amp_max_db,
        )

    def next_tier_name(self, textgrid: TextGrid, current_tier_name: str | None) -> str | None:
        if not textgrid.tiers:
            return None
        if not current_tier_name:
            return textgrid.tiers[0].name
        names = [tier.name for tier in textgrid.tiers]
        if current_tier_name not in names:
            return names[0]
        index = names.index(current_tier_name)
        return names[(index + 1) % len(names)]

    def extract_label_in_range(
        self,
        textgrid: TextGrid | None,
        tier_name: str | None,
        start_sec: float,
        end_sec: float,
    ) -> str:
        if textgrid is None or not tier_name:
            return ""
        tier = next((tier for tier in textgrid.tiers if tier.name == tier_name), None)
        if tier is None:
            return ""
        labels: list[str] = []
        for interval in tier.intervals:
            if interval.xmax <= start_sec or interval.xmin >= end_sec:
                continue
            if (
                start_sec < end_sec
                and interval.xmin < end_sec <= interval.xmax
                and not (interval.xmin <= start_sec < interval.xmax)
            ):
                continue
            label = interval.text.strip()
            if not label:
                continue
            if label not in labels:
                labels.append(label)
        return "+".join(labels)

    def save_plot_figure(
        self,
        fig: matplotlib.figure.Figure,
        output_dir: Path,
        wav_stem: str,
        textgrid_label: str,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = self._sanitize_filename(wav_stem)
        safe_label = self._sanitize_filename(textgrid_label)
        if safe_label:
            name = f"LPC_Spectrum_{safe_stem}_{safe_label}.png"
        else:
            name = f"LPC_Spectrum_{safe_stem}.png"
        output_path = output_dir / name
        fig.savefig(output_path, dpi=300)
        return output_path

    def _sanitize_filename(self, value: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value).strip()
        cleaned = re.sub(r"\s+", "_", cleaned)
        return cleaned[:120]
