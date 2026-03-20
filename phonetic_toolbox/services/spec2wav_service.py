import numpy as np
import soundfile as sf
import os
from typing import Optional, Tuple
from scipy import signal

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
        
        # Window length usually equals n_fft
        window_length = n_fft
        
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
        
        # 4. Compute reconstructed spectrogram for verification (optional but good for UI)
        # We compute this at the native SR or Target SR? 
        # The GUI showed the reconstructed one. Let's compute it.
        # We'll use a standard window and n_fft appropriate for the audio.
        # But wait, to compare with original, we might want similar params.
        # However, the original params were derived from image dimensions.
        # Let's just compute a standard spectrogram of the output audio.
        
        # Recalculate STFT of the result
        # Use consistent n_fft if possible, or standard one
        calc_n_fft = n_fft # Use same resolution
        calc_win_length = min(1024, calc_n_fft)
        calc_window = signal.windows.hann(calc_win_length, sym=False)
        
        # If we resampled, the hop_length and n_fft relationship to time changed.
        # But we can just compute a standard STFT for display.
        if sr != native_sr:
             # Recalculate hop_length for the new SR to maintain similar time resolution?
             # Or just standard values.
             # Let's stick to the parameters used in generation if not resampled, 
             # otherwise standard.
             pass

        # To be safe and simple, let's just compute a standard spectrogram for the output audio
        # using the values that fit the current audio.
        # But the GUI implementation used the SAME n_fft and hop_length (which might be wrong if resampled?)
        # The GUI implementation:
        # reconstructed_spec = np.abs(_stft(native_audio, n_fft, hop_length, window_length, window))
        # It used 'native_audio' BEFORE resampling for the reconstruction check.
        
        # Let's follow that logic: Compute reconstruction check on NATIVE audio
        reconstructed_spec = np.abs(_stft(native_audio, n_fft, hop_length, min(window_length, n_fft), signal.windows.hann(min(window_length, n_fft), sym=False)))
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
