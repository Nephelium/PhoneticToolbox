import numpy as np
import soundfile as sf
import os

from ..core.spec2wav.image_processing import load_spectrogram_image
from ..core.spec2wav.griffin_lim import spectrogram_to_audio, _stft
from ..core.spec2wav.common import resample_audio, amplitude_to_db
from ..models.spec2wav_models import Spec2WavConfig, Spec2WavResult

class Spec2WavService:
    """
    Service for converting spectrogram images to audio waveforms.
    """
    
    def convert(self, config: Spec2WavConfig) -> Spec2WavResult:
        """
        Convert a spectrogram image to audio based on the provided configuration.
        
        Args:
            config (Spec2WavConfig): Configuration object containing image path/data and parameters.
            
        Returns:
            Spec2WavResult: Object containing the reconstructed audio and metadata.
            
        Raises:
            ValueError: If image loading or processing fails.
        """
        self._validate_config(config)
        if config.image_path and not os.path.exists(config.image_path) and config.image_data is None:
            raise ValueError(f"Image file not found: {config.image_path}")
            
        # 1. Load and process image
        img, linear_spectrogram, log_spectrogram, hop_length, n_fft = load_spectrogram_image(
            image_path=config.image_path,
            image_data=config.image_data,
            time_end=config.time_end,
            freq_end=config.freq_end,
            time_start=config.time_start,
            freq_start=config.freq_start,
            min_dB=config.min_db,
            max_dB=config.max_db
        )
        
        # 2. Reconstruct audio using Griffin-Lim
        # Calculate native sample rate based on frequency range (Nyquist)
        # Usually sr = 2 * max_freq
        native_sr = int(2 * config.freq_end)
        
        # Calculate window length in samples
        if hasattr(config, 'win_length_ms') and config.win_length_ms > 0:
            window_length = int((config.win_length_ms / 1000.0) * native_sr)
            # Ensure window length is not greater than n_fft
            window_length = min(window_length, n_fft)
        else:
            window_length = n_fft
        window_length = max(1, window_length)
        
        native_audio = spectrogram_to_audio(
            spectrogram=linear_spectrogram,
            hop_length=hop_length,
            window_length=window_length,
            n_fft=n_fft,
            sr=native_sr,
            n_iter=config.n_iter
        )
        
        # 3. Resample if needed
        if config.target_sr > 0 and config.target_sr != native_sr:
            audio = resample_audio(native_audio, native_sr, config.target_sr)
            sr = config.target_sr
        else:
            audio = native_audio
            sr = native_sr
        
        # Calculate actual duration
        duration = len(audio) / sr
        
        reconstructed_spec = np.abs(_stft(native_audio, n_fft, hop_length, min(window_length, n_fft), np.hanning(min(window_length, n_fft))))
        reconstructed_db = amplitude_to_db(reconstructed_spec, ref=np.max)

        return Spec2WavResult(
            audio=audio,
            sr=sr,
            spectrogram=linear_spectrogram,
            log_spectrogram=log_spectrogram,
            reconstructed_spectrogram_db=reconstructed_db,
            duration=duration
        )
    
    def save_audio(self, audio: np.ndarray, sr: int, file_path: str):
        """
        Save the reconstructed audio to a WAV file.
        
        Args:
            audio (np.ndarray): Audio data.
            sr (int): Sample rate.
            file_path (str): Output file path.
        """
        sf.write(file_path, audio, sr)

    @staticmethod
    def _validate_config(config: Spec2WavConfig) -> None:
        if config.image_data is None and not config.image_path:
            raise ValueError("Either image_path or image_data must be provided.")
        if config.time_end <= config.time_start:
            raise ValueError("time_end must be greater than time_start.")
        if config.freq_start < 0:
            raise ValueError("freq_start must be non-negative.")
        if config.freq_end <= config.freq_start:
            raise ValueError("freq_end must be greater than freq_start.")
        if config.min_db >= config.max_db:
            raise ValueError("min_db must be less than max_db.")
        if config.n_iter < 1:
            raise ValueError("n_iter must be at least 1.")
        if config.target_sr < 0:
            raise ValueError("target_sr must be non-negative.")
