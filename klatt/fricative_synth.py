"""
fricative_synth.py

Provides fricative consonant synthesis using bandpass-filtered white noise.
Supports both voiceless and voiced fricatives with appropriate noise
characteristics based on place of articulation.

Classes:
    FricativeSynthesizer: Synthesizes fricative consonants

Requirements:
    - 4.1: Sibilant fricatives use high-frequency friction noise
    - 4.2: Non-sibilant fricatives use appropriate center frequency noise
    - 4.3: Noise center frequency based on place of articulation
    - 4.4: Voiceless fricatives have AV=0, only friction noise (AF>0)
    - 4.5: Voiced fricatives have both voicing (AV>0) and friction noise (AF>0)
    - 4.6: /h/ uses whisper phonation type parameters
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Optional, Dict

from klatt.consonant_data import ConsonantParams, CONSONANT_DATA


class FricativeSynthesizer:
    """
    Synthesizes fricative consonants using bandpass-filtered white noise.
    
    Fricatives are produced by turbulent airflow through a constriction.
    The acoustic characteristics depend on:
    - Place of articulation (determines noise center frequency)
    - Voicing (voiced fricatives have periodic voicing + noise)
    
    Attributes:
        fs: Sampling frequency in Hz
        f0: Default fundamental frequency for voiced fricatives
    """
    
    def __init__(self, fs: int = 16000, f0: float = 120.0):
        """
        Initialize the fricative synthesizer.
        
        Args:
            fs: Sampling frequency in Hz (default 16000)
            f0: Default fundamental frequency for voiced fricatives (default 120 Hz)
        """
        self.fs = fs
        self.f0 = f0
    
    def synthesize(self, params: ConsonantParams, duration_ms: float,
                   context: Optional[Dict] = None) -> np.ndarray:
        """
        Synthesize a fricative consonant.
        
        Args:
            params: ConsonantParams object containing acoustic parameters
            duration_ms: Duration of the fricative in milliseconds
            context: Optional context information (preceding/following segments)
            
        Returns:
            Audio waveform as numpy array (normalized to [-1, 1])
        """
        n_samples = int(duration_ms * self.fs / 1000)
        
        if n_samples <= 0:
            return np.array([])
        
        # Special handling for /h/ - use whisper synthesis
        if params.symbol == 'h':
            return self._synthesize_h(n_samples, params)
        
        # Generate friction noise
        friction_noise = self._generate_friction_noise(n_samples, params)
        
        # For voiced fricatives, add voicing component
        if params.voiced:
            voicing = self._generate_voicing(n_samples, params)
            # Mix voicing and noise (typical ratio: 30% voicing, 70% noise)
            output = 0.3 * voicing + 0.7 * friction_noise
        else:
            # Voiceless fricative: only friction noise
            output = friction_noise
        
        return self._normalize(output)
    
    def _generate_friction_noise(self, n_samples: int, 
                                  params: ConsonantParams) -> np.ndarray:
        """
        Generate bandpass-filtered white noise for friction.
        
        The center frequency and bandwidth are determined by the
        place of articulation stored in the ConsonantParams.
        
        Args:
            n_samples: Number of samples to generate
            params: ConsonantParams with noise_freq and noise_bw
            
        Returns:
            Filtered noise array
        """
        # Generate white noise
        noise = np.random.randn(n_samples)
        
        # Get center frequency and bandwidth from params
        center_freq = params.noise_freq
        bandwidth = params.noise_bw
        
        # Ensure valid frequency range for filter design
        nyquist = self.fs / 2
        
        # Calculate low and high cutoff frequencies
        low_freq = max(center_freq - bandwidth / 2, 50)  # Minimum 50 Hz
        high_freq = min(center_freq + bandwidth / 2, nyquist - 100)  # Leave margin
        
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
    
    def _generate_voicing(self, n_samples: int, 
                          params: ConsonantParams) -> np.ndarray:
        """
        Generate a simple voicing waveform for voiced fricatives.
        
        Uses a simple impulse train filtered through a low-pass filter
        to approximate glottal pulses.
        
        Args:
            n_samples: Number of samples to generate
            params: ConsonantParams (used for potential F0 variation)
            
        Returns:
            Voicing waveform array
        """
        # Calculate period in samples
        period_samples = int(self.fs / self.f0)
        
        if period_samples <= 0:
            return np.zeros(n_samples)
        
        # Generate impulse train
        impulse_train = np.zeros(n_samples)
        for i in range(0, n_samples, period_samples):
            impulse_train[i] = 1.0
        
        # Apply simple lowpass filtering to smooth the impulses
        # This creates a more natural glottal pulse shape
        nyquist = self.fs / 2
        cutoff = min(1000, nyquist - 100) / nyquist  # 1000 Hz lowpass
        
        try:
            b, a = butter(2, cutoff, btype='low')
            voicing = filtfilt(b, a, impulse_train)
        except ValueError:
            voicing = impulse_train
        
        return voicing
    
    def _synthesize_h(self, n_samples: int, params: ConsonantParams) -> np.ndarray:
        """
        Synthesize /h/ using whisper-like parameters.
        
        /h/ is characterized by:
        - Wide-band noise (aspiration)
        - No periodic voicing (voiceless)
        - Formant structure similar to following vowel (if context available)
        
        Requirements: 4.6 - Use "whisper" phonation type parameters
        
        Args:
            n_samples: Number of samples to generate
            params: ConsonantParams for /h/
            
        Returns:
            Whisper-like audio waveform
        """
        # Generate wide-band noise (characteristic of whisper/aspiration)
        noise = np.random.randn(n_samples)
        
        # Apply gentle high-pass filter to remove very low frequencies
        # Whisper has less low-frequency energy
        nyquist = self.fs / 2
        highpass_cutoff = min(200, nyquist - 100) / nyquist
        
        try:
            b, a = butter(2, highpass_cutoff, btype='high')
            filtered = filtfilt(b, a, noise)
        except ValueError:
            filtered = noise
        
        # Apply amplitude envelope (slight attack and decay)
        envelope = self._create_envelope(n_samples, attack_ratio=0.1, decay_ratio=0.1)
        output = filtered * envelope
        
        return self._normalize(output)
    
    def _create_envelope(self, n_samples: int, 
                         attack_ratio: float = 0.05,
                         decay_ratio: float = 0.05) -> np.ndarray:
        """
        Create an amplitude envelope with attack and decay.
        
        Args:
            n_samples: Total number of samples
            attack_ratio: Proportion of duration for attack (0-1)
            decay_ratio: Proportion of duration for decay (0-1)
            
        Returns:
            Envelope array (values 0-1)
        """
        envelope = np.ones(n_samples)
        
        # Attack phase
        attack_samples = int(n_samples * attack_ratio)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        decay_samples = int(n_samples * decay_ratio)
        if decay_samples > 0:
            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        return envelope
    
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
    
    def get_noise_frequency(self, symbol: str) -> float:
        """
        Get the noise center frequency for a fricative symbol.
        
        Args:
            symbol: IPA fricative symbol
            
        Returns:
            Center frequency in Hz, or 0 if not a fricative
        """
        if symbol in CONSONANT_DATA:
            return CONSONANT_DATA[symbol].noise_freq
        return 0.0
    
    def is_voiced(self, symbol: str) -> bool:
        """
        Check if a fricative symbol is voiced.
        
        Args:
            symbol: IPA fricative symbol
            
        Returns:
            True if voiced, False otherwise
        """
        if symbol in CONSONANT_DATA:
            return CONSONANT_DATA[symbol].voiced
        return False
