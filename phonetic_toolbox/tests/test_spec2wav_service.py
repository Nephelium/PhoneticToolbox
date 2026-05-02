import numpy as np
import pytest

from phonetic_toolbox.core.spec2wav.image_processing import load_spectrogram_image
from phonetic_toolbox.models.spec2wav_models import Spec2WavConfig
from phonetic_toolbox.services.spec2wav_service import Spec2WavService


def test_spec2wav_rejects_invalid_frequency_range():
    config = Spec2WavConfig(
        image_data=np.ones((8, 8), dtype=np.uint8),
        freq_start=5000,
        freq_end=5000,
    )

    with pytest.raises(ValueError, match="freq_end"):
        Spec2WavService().convert(config)


def test_spec2wav_rejects_invalid_db_range():
    config = Spec2WavConfig(
        image_data=np.ones((8, 8), dtype=np.uint8),
        min_db=0,
        max_db=-30,
    )

    with pytest.raises(ValueError, match="min_db"):
        Spec2WavService().convert(config)


def test_load_spectrogram_image_rejects_empty_image_data():
    with pytest.raises(ValueError, match="empty"):
        load_spectrogram_image(image_data=np.array([], dtype=np.uint8))
