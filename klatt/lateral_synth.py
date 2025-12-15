"""
lateral_synth.py

Provides lateral consonant synthesis supporting lateral fricatives,
lateral approximants, and lateral flaps.

Classes:
    LateralSynthesizer: Synthesizes lateral consonants

Requirements:
    - 7.1: Synthesize lateral fricatives (ɬ, ɮ) with lateral formants and friction noise
    - 7.2: Synthesize lateral approximants (l, ɭ, ʎ, ʟ) with lateral formant parameters
    - 7.3: Synthesize lateral flaps (ɺ) as short lateral sounds
    - 7.4: Set characteristic lateral formant patterns
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from typing import Optional, Dict
from math import gcd

from klatt.consonant_data import (
    ConsonantParams, CONSONANT_DATA, 
    LATERAL_FRICATIVES, LATERAL_APPROXIMANTS, LATERAL_FLAPS
)
from klatt.tdklatt import KlattParam1980, klatt_make


class LateralSynthesizer:
    """
    Synthesizes lateral consonants using the Klatt formant synthesizer.
    
    Laterals are produced with airflow around the sides of the tongue.
    Three types are supported:
    
    1. Lateral Approximants (l, ɭ, ʎ, ʟ):
       - Vowel-like formant structure
       - All voiced (AV > 0)
       - Characteristic low F2 and high F3
    
    2. Lateral Fricatives (ɬ, ɮ):
       - Formant structure plus friction noise
       - ɬ is voiceless (AV = 0, AF > 0)
       - ɮ is voiced (AV > 0, AF > 0)
    
    3. Lateral Flaps (ɺ):
       - Very short duration (~30ms)
       - Voiced, vowel-like
    
    Attributes:
        fs: Target sampling frequency in Hz
        f0: Default fundamental frequency for voicing
        klatt_fs: Internal Klatt synthesizer sampling rate (10000 Hz)
    """
    
    # Klatt synthesizer internal sample rate
    KLATT_FS = 10000
    
    # Default parameters
    DEFAULT_AV = 60      # Voicing amplitude (dB)
    DEFAULT_F0 = 120.0   # Default fundamental frequency (Hz)
    
    def __init__(self, fs: int = 16000, f0: float = 120.0):
        """
        Initialize the lateral synthesizer.
        
        Args:
            fs: Target output sampling frequency in Hz (default 16000)
            f0: Default fundamental frequency for voicing (default 120 Hz)
        """
        self.fs = fs
        self.f0 = f0
        self.klatt_fs = self.KLATT_FS
    
    def synthesize(self, params: ConsonantParams, duration_ms: float,
                   context: Optional[Dict] = None) -> np.ndarray:
        """
        Synthesize a lateral consonant.
        
        Routes to appropriate synthesis method based on manner of articulation.
        
        Args:
            params: ConsonantParams object containing acoustic parameters
            duration_ms: Duration of the lateral in milliseconds
            context: Optional context information (preceding/following segments)
            
        Returns:
            Audio waveform as numpy array (normalized to [-1, 1])
        """
        if duration_ms <= 0:
            return np.array([])
        
        manner = params.manner
        
        if manner == 'lateral_fricative':
            return self._synthesize_lateral_fricative(params, duration_ms, context)
        elif manner == 'lateral_approximant':
            return self._synthesize_lateral_approximant(params, duration_ms, context)
        elif manner == 'lateral_flap':
            return self._synthesize_lateral_flap(params, duration_ms, context)
        else:
            # Default to approximant-like synthesis
            return self._synthesize_lateral_approximant(params, duration_ms, context)

    
    def _synthesize_lateral_approximant(self, params: ConsonantParams, 
                                         duration_ms: float,
                                         context: Optional[Dict] = None) -> np.ndarray:
        """
        Synthesize a lateral approximant (l, ɭ, ʎ, ʟ).
        
        Uses the Klatt synthesizer with formant parameters.
        All lateral approximants are voiced.
        
        Requirements: 7.2, 7.4
        
        Args:
            params: ConsonantParams with lateral acoustic parameters
            duration_ms: Duration in milliseconds
            context: Optional context information
            
        Returns:
            Audio waveform as numpy array
        """
        # Create Klatt parameters
        klatt_params = self._create_klatt_params(params, duration_ms, voiced=True)
        
        # Run the Klatt synthesizer
        synth = klatt_make(klatt_params)
        synth.run()
        
        # Resample to target sample rate
        output = self._resample_output(synth.output)
        
        return self._normalize(output)
    
    def _synthesize_lateral_fricative(self, params: ConsonantParams,
                                       duration_ms: float,
                                       context: Optional[Dict] = None) -> np.ndarray:
        """
        Synthesize a lateral fricative (ɬ, ɮ).
        
        Combines formant synthesis with friction noise.
        ɬ is voiceless, ɮ is voiced.
        
        Requirements: 7.1, 7.4
        
        Args:
            params: ConsonantParams with lateral fricative parameters
            duration_ms: Duration in milliseconds
            context: Optional context information
            
        Returns:
            Audio waveform as numpy array
        """
        n_samples = int(duration_ms * self.fs / 1000)
        
        if n_samples <= 0:
            return np.array([])
        
        # Generate friction noise component
        friction_noise = self._generate_friction_noise(n_samples, params)
        
        if params.voiced:
            # Voiced lateral fricative (ɮ): formants + noise
            # Create Klatt parameters for voiced component
            klatt_params = self._create_klatt_params(params, duration_ms, voiced=True)
            synth = klatt_make(klatt_params)
            synth.run()
            voicing = self._resample_output(synth.output)
            
            # Ensure same length
            min_len = min(len(voicing), len(friction_noise))
            voicing = voicing[:min_len]
            friction_noise = friction_noise[:min_len]
            
            # Mix voicing and noise (typical ratio: 40% voicing, 60% noise)
            output = 0.4 * voicing + 0.6 * friction_noise
        else:
            # Voiceless lateral fricative (ɬ): only friction noise
            output = friction_noise
        
        return self._normalize(output)
    
    def _synthesize_lateral_flap(self, params: ConsonantParams,
                                  duration_ms: float,
                                  context: Optional[Dict] = None) -> np.ndarray:
        """
        Synthesize a lateral flap (ɺ).
        
        Very short duration lateral sound, similar to lateral approximant
        but with rapid onset and offset.
        
        Requirements: 7.3
        
        Args:
            params: ConsonantParams with lateral flap parameters
            duration_ms: Duration in milliseconds (typically ~30ms)
            context: Optional context information
            
        Returns:
            Audio waveform as numpy array
        """
        # Lateral flaps are very short - use the default duration from params
        # which is typically 30ms
        actual_duration = min(duration_ms, params.default_duration)
        
        # Create Klatt parameters
        klatt_params = self._create_klatt_params(params, actual_duration, voiced=True)
        
        # Run the Klatt synthesizer
        synth = klatt_make(klatt_params)
        synth.run()
        
        # Resample to target sample rate
        output = self._resample_output(synth.output)
        
        # Apply rapid attack/decay envelope for flap characteristic
        output = self._apply_flap_envelope(output)
        
        return self._normalize(output)
    
    def _create_klatt_params(self, params: ConsonantParams, 
                              duration_ms: float,
                              voiced: bool = True) -> KlattParam1980:
        """
        Create Klatt synthesizer parameters for lateral synthesis.
        
        Sets up:
        - Formants F1-F3 based on place of articulation
        - Voicing amplitude based on voiced parameter
        
        Args:
            params: ConsonantParams with lateral acoustic parameters
            duration_ms: Duration in milliseconds
            voiced: Whether to include voicing
            
        Returns:
            KlattParam1980 object configured for lateral synthesis
        """
        duration_sec = duration_ms / 1000.0
        
        # Set voicing amplitude
        av = self.DEFAULT_AV if voiced else 0
        
        # Create Klatt parameters
        klatt_params = KlattParam1980(
            FS=self.klatt_fs,
            DUR=duration_sec,
            F0=self.f0,
            # Voicing
            AV=av,
            AVS=0,
            AH=0,
            AF=0,
            # No nasal pole/zero for laterals
            FNP=250,
            BNP=100,
            FNZ=250,
            BNZ=100,
            # Use cascade synthesis (SW=0)
            SW=0,
            # Formants based on place of articulation (Requirement 7.4)
            FF=[params.f1, params.f2, params.f3, 3500, 4500],
            BW=[80, 100, 120, 200, 250],
        )
        
        return klatt_params

    
    def _generate_friction_noise(self, n_samples: int, 
                                  params: ConsonantParams) -> np.ndarray:
        """
        Generate bandpass-filtered white noise for lateral fricatives.
        
        Args:
            n_samples: Number of samples to generate
            params: ConsonantParams with noise_freq and noise_bw
            
        Returns:
            Filtered noise array
        """
        # Generate white noise
        noise = np.random.randn(n_samples)
        
        # Get center frequency and bandwidth from params
        center_freq = params.noise_freq if params.noise_freq > 0 else 4000
        bandwidth = params.noise_bw if params.noise_bw > 0 else 1500
        
        # Ensure valid frequency range for filter design
        nyquist = self.fs / 2
        
        # Calculate low and high cutoff frequencies
        low_freq = max(center_freq - bandwidth / 2, 50)
        high_freq = min(center_freq + bandwidth / 2, nyquist - 100)
        
        # Normalize frequencies for filter design
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Ensure valid normalized frequencies (0 < freq < 1)
        low_norm = max(0.01, min(low_norm, 0.98))
        high_norm = max(low_norm + 0.01, min(high_norm, 0.99))
        
        # Design and apply bandpass filter
        try:
            b, a = butter(4, [low_norm, high_norm], btype='band')
            filtered_noise = filtfilt(b, a, noise)
        except ValueError:
            # Fallback: return unfiltered noise if filter design fails
            filtered_noise = noise
        
        return filtered_noise
    
    def _apply_flap_envelope(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply rapid attack/decay envelope for flap sounds.
        
        Flaps have very rapid onset and offset, creating a
        brief "tap" sound.
        
        Args:
            audio: Input audio array
            
        Returns:
            Audio with flap envelope applied
        """
        if len(audio) == 0:
            return audio
        
        n_samples = len(audio)
        
        # Very rapid attack (5% of duration)
        attack_samples = max(1, int(n_samples * 0.05))
        # Rapid decay (10% of duration)
        decay_samples = max(1, int(n_samples * 0.10))
        
        envelope = np.ones(n_samples)
        
        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        if decay_samples > 0:
            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        return audio * envelope
    
    def _resample_output(self, audio: np.ndarray) -> np.ndarray:
        """
        Resample Klatt output to target sample rate.
        
        Args:
            audio: Audio from Klatt synthesizer at 10000 Hz
            
        Returns:
            Resampled audio at target sample rate
        """
        if len(audio) == 0:
            return audio
        
        if self.fs == self.klatt_fs:
            return audio
        
        # Use rational resampling
        up = self.fs
        down = self.klatt_fs
        g = gcd(up, down)
        up //= g
        down //= g
        
        return resample_poly(audio, up, down)
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        if len(audio) == 0:
            return audio
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def get_formants(self, symbol: str) -> tuple:
        """
        Get the formant frequencies for a lateral symbol.
        
        Args:
            symbol: IPA lateral symbol
            
        Returns:
            Tuple of (F1, F2, F3) in Hz, or (0, 0, 0) if not a lateral
        """
        if symbol in CONSONANT_DATA:
            params = CONSONANT_DATA[symbol]
            return (params.f1, params.f2, params.f3)
        return (0.0, 0.0, 0.0)
    
    def is_lateral(self, symbol: str) -> bool:
        """
        Check if a symbol is a lateral consonant.
        
        Args:
            symbol: IPA symbol
            
        Returns:
            True if lateral, False otherwise
        """
        return symbol in (LATERAL_FRICATIVES | LATERAL_APPROXIMANTS | LATERAL_FLAPS)
    
    def is_lateral_fricative(self, symbol: str) -> bool:
        """
        Check if a symbol is a lateral fricative.
        
        Args:
            symbol: IPA symbol
            
        Returns:
            True if lateral fricative, False otherwise
        """
        return symbol in LATERAL_FRICATIVES
    
    def is_lateral_approximant(self, symbol: str) -> bool:
        """
        Check if a symbol is a lateral approximant.
        
        Args:
            symbol: IPA symbol
            
        Returns:
            True if lateral approximant, False otherwise
        """
        return symbol in LATERAL_APPROXIMANTS
    
    def is_lateral_flap(self, symbol: str) -> bool:
        """
        Check if a symbol is a lateral flap.
        
        Args:
            symbol: IPA symbol
            
        Returns:
            True if lateral flap, False otherwise
        """
        return symbol in LATERAL_FLAPS
    
    def is_voiced(self, symbol: str) -> bool:
        """
        Check if a lateral symbol is voiced.
        
        Args:
            symbol: IPA lateral symbol
            
        Returns:
            True if voiced, False otherwise
        """
        if symbol in CONSONANT_DATA:
            return CONSONANT_DATA[symbol].voiced
        return False
