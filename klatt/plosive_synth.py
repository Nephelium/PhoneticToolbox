"""
plosive_synth.py

Provides plosive (stop) consonant synthesis with closure and release phases.
Supports both voiceless and voiced plosives with appropriate acoustic
characteristics based on place of articulation.

Classes:
    PlosiveSynthesizer: Synthesizes plosive consonants

Requirements:
    - 3.1: Synthesize plosives with closure-release characteristics
    - 3.2: Plosive duration is very short (closure ~50-100ms, release ~10-30ms)
    - 3.3: Voiceless plosives have AV=0 during closure
    - 3.4: Voiced plosives maintain low-amplitude voicing during closure
    - 3.5: Release burst with noise and formant transition
    - 3.6: Aspirated plosives have increased AH parameter in release
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Optional, Dict

from klatt.consonant_data import ConsonantParams, CONSONANT_DATA, PLOSIVES


class PlosiveSynthesizer:
    """
    Synthesizes plosive (stop) consonants.
    
    Plosives are produced by complete closure of the vocal tract followed
    by a sudden release. The acoustic characteristics depend on:
    - Place of articulation (determines burst frequency characteristics)
    - Voicing (voiced plosives have low-amplitude voicing during closure)
    - Aspiration (aspirated plosives have noise after release)
    
    Plosive structure:
    1. Closure phase: Complete obstruction (silence or low voicing)
    2. Release phase: Burst noise with exponential decay
    
    Attributes:
        fs: Sampling frequency in Hz
        f0: Default fundamental frequency for voiced plosives
        closure_duration_ms: Duration of closure phase in ms
        release_duration_ms: Duration of release burst in ms
        min_total_duration_ms: Minimum total plosive duration
        max_total_duration_ms: Maximum total plosive duration
    """
    
    # Duration constraints for plosives (Requirements 3.2, 8.2)
    MIN_TOTAL_DURATION_MS = 50.0
    MAX_TOTAL_DURATION_MS = 130.0
    DEFAULT_CLOSURE_MS = 60.0
    DEFAULT_RELEASE_MS = 20.0
    
    def __init__(self, fs: int = 16000, f0: float = 120.0):
        """
        Initialize the plosive synthesizer.
        
        Args:
            fs: Sampling frequency in Hz (default 16000)
            f0: Default fundamental frequency for voiced plosives (default 120 Hz)
        """
        self.fs = fs
        self.f0 = f0
        self.closure_duration_ms = self.DEFAULT_CLOSURE_MS
        self.release_duration_ms = self.DEFAULT_RELEASE_MS
    
    def synthesize(self, params: ConsonantParams, duration_ms: float,
                   context: Optional[Dict] = None,
                   aspirated: bool = False) -> np.ndarray:
        """
        Synthesize a plosive consonant.
        
        Plosives have fixed duration regardless of duration_ms parameter
        (Requirements 3.2, 8.2). The duration is constrained to 50-130ms.
        
        Args:
            params: ConsonantParams object containing acoustic parameters
            duration_ms: Requested duration (ignored for plosives - fixed duration)
            context: Optional context information (preceding/following segments)
            aspirated: Whether to add aspiration noise after release (Req 3.6)
            
        Returns:
            Audio waveform as numpy array (normalized to [-1, 1])
        """
        # Plosives have fixed duration - constrain to valid range
        total_duration = self._constrain_duration(params.default_duration)
        
        # Calculate closure and release durations
        closure_ms, release_ms = self._calculate_phase_durations(total_duration)
        
        # Generate closure phase
        closure = self._generate_closure(closure_ms, params)
        
        # Generate release burst
        release = self._generate_release(release_ms, params, aspirated)
        
        # Concatenate phases
        output = np.concatenate([closure, release])
        
        return self._normalize(output)
    
    def _constrain_duration(self, duration_ms: float) -> float:
        """
        Constrain plosive duration to valid range.
        
        Requirements 3.2, 8.2: Plosive duration is fixed and cannot be
        adjusted by duration modifiers.
        
        Args:
            duration_ms: Requested duration
            
        Returns:
            Constrained duration within [MIN_TOTAL_DURATION_MS, MAX_TOTAL_DURATION_MS]
        """
        return max(self.MIN_TOTAL_DURATION_MS, 
                   min(duration_ms, self.MAX_TOTAL_DURATION_MS))
    
    def _calculate_phase_durations(self, total_ms: float) -> tuple:
        """
        Calculate closure and release phase durations.
        
        Maintains approximately 75% closure, 25% release ratio.
        
        Args:
            total_ms: Total plosive duration in ms
            
        Returns:
            Tuple of (closure_ms, release_ms)
        """
        # Typical ratio: ~75% closure, ~25% release
        closure_ratio = 0.75
        closure_ms = total_ms * closure_ratio
        release_ms = total_ms * (1 - closure_ratio)
        
        # Ensure minimum release duration for audible burst
        min_release = 10.0
        if release_ms < min_release:
            release_ms = min_release
            closure_ms = total_ms - release_ms
        
        return closure_ms, release_ms
    
    def _generate_closure(self, duration_ms: float, 
                          params: ConsonantParams) -> np.ndarray:
        """
        Generate the closure phase of a plosive.
        
        Requirements:
        - 3.3: Voiceless plosives have AV=0 (silence)
        - 3.4: Voiced plosives have low-amplitude voicing
        
        Args:
            duration_ms: Duration of closure in ms
            params: ConsonantParams for the plosive
            
        Returns:
            Closure phase audio array
        """
        n_samples = int(duration_ms * self.fs / 1000)
        
        if n_samples <= 0:
            return np.array([])
        
        if params.voiced:
            # Voiced plosive: low-amplitude voicing during closure (Req 3.4)
            return self._generate_low_voicing(n_samples, params)
        else:
            # Voiceless plosive: silence during closure (Req 3.3)
            return np.zeros(n_samples)
    
    def _generate_low_voicing(self, n_samples: int, 
                               params: ConsonantParams) -> np.ndarray:
        """
        Generate low-amplitude voicing for voiced plosive closure.
        
        Creates a muffled, low-amplitude periodic signal to simulate
        the "voice bar" heard during voiced stop closures.
        
        Args:
            n_samples: Number of samples to generate
            params: ConsonantParams for the plosive
            
        Returns:
            Low-amplitude voicing array
        """
        if n_samples <= 0:
            return np.array([])
        
        # Calculate period in samples
        period_samples = int(self.fs / self.f0)
        
        if period_samples <= 0:
            return np.zeros(n_samples)
        
        # Generate impulse train
        impulse_train = np.zeros(n_samples)
        for i in range(0, n_samples, period_samples):
            impulse_train[i] = 1.0
        
        # Apply heavy lowpass filtering to create muffled voicing
        # Voice bar is characterized by very low frequency energy
        nyquist = self.fs / 2
        cutoff = min(300, nyquist - 100) / nyquist  # Very low cutoff
        
        try:
            b, a = butter(2, cutoff, btype='low')
            voicing = filtfilt(b, a, impulse_train)
        except ValueError:
            voicing = impulse_train * 0.1
        
        # Scale to low amplitude (voice bar is quiet)
        voicing = voicing * 0.15
        
        return voicing
    
    def _generate_release(self, duration_ms: float, params: ConsonantParams,
                          aspirated: bool = False) -> np.ndarray:
        """
        Generate the release burst of a plosive.
        
        Requirements:
        - 3.5: Release burst with noise and formant transition
        - 3.6: Aspirated plosives have increased AH (aspiration noise)
        
        Args:
            duration_ms: Duration of release in ms
            params: ConsonantParams for the plosive
            aspirated: Whether to add aspiration noise
            
        Returns:
            Release burst audio array
        """
        n_samples = int(duration_ms * self.fs / 1000)
        
        if n_samples <= 0:
            return np.array([])
        
        # Generate burst noise
        burst = self._generate_burst(n_samples, params)
        
        # Add aspiration if requested (Req 3.6)
        if aspirated:
            aspiration = self._generate_aspiration(n_samples)
            # Mix burst and aspiration
            burst = 0.6 * burst + 0.4 * aspiration
        
        return burst
    
    def _generate_burst(self, n_samples: int, 
                        params: ConsonantParams) -> np.ndarray:
        """
        Generate release burst noise.
        
        The burst characteristics depend on place of articulation:
        - Bilabial: Low-frequency burst (~500 Hz)
        - Alveolar: High-frequency burst (~4000 Hz)
        - Velar: Mid-frequency burst (~2000 Hz)
        
        Args:
            n_samples: Number of samples to generate
            params: ConsonantParams with noise_freq and noise_bw
            
        Returns:
            Burst noise array with exponential decay
        """
        if n_samples <= 0:
            return np.array([])
        
        # Generate white noise
        noise = np.random.randn(n_samples)
        
        # Get burst frequency characteristics from params
        center_freq = params.noise_freq if params.noise_freq > 0 else 2000
        bandwidth = params.noise_bw if params.noise_bw > 0 else 1000
        
        # Apply bandpass filter centered on burst frequency
        nyquist = self.fs / 2
        low_freq = max(center_freq - bandwidth / 2, 50)
        high_freq = min(center_freq + bandwidth / 2, nyquist - 100)
        
        low_norm = max(0.01, min(low_freq / nyquist, 0.98))
        high_norm = max(low_norm + 0.01, min(high_freq / nyquist, 0.99))
        
        try:
            b, a = butter(4, [low_norm, high_norm], btype='band')
            filtered_noise = filtfilt(b, a, noise)
        except ValueError:
            filtered_noise = noise
        
        # Apply exponential decay envelope (burst decays rapidly)
        decay_rate = 5.0  # Controls how fast the burst decays
        envelope = np.exp(-np.linspace(0, decay_rate, n_samples))
        
        return filtered_noise * envelope
    
    def _generate_aspiration(self, n_samples: int) -> np.ndarray:
        """
        Generate aspiration noise for aspirated plosives.
        
        Requirement 3.6: Aspirated plosives have increased AH parameter.
        Aspiration is wide-band noise similar to /h/.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Aspiration noise array
        """
        if n_samples <= 0:
            return np.array([])
        
        # Generate wide-band noise
        noise = np.random.randn(n_samples)
        
        # Apply gentle highpass to remove very low frequencies
        nyquist = self.fs / 2
        highpass_cutoff = min(200, nyquist - 100) / nyquist
        
        try:
            b, a = butter(2, highpass_cutoff, btype='high')
            filtered = filtfilt(b, a, noise)
        except ValueError:
            filtered = noise
        
        # Apply gradual decay envelope
        envelope = np.exp(-np.linspace(0, 3, n_samples))
        
        return filtered * envelope
    
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
    
    def get_total_duration(self, params: ConsonantParams) -> float:
        """
        Get the actual total duration for a plosive.
        
        Plosives have fixed duration regardless of modifiers.
        
        Args:
            params: ConsonantParams for the plosive
            
        Returns:
            Total duration in milliseconds
        """
        return self._constrain_duration(params.default_duration)
    
    def is_voiced(self, symbol: str) -> bool:
        """
        Check if a plosive symbol is voiced.
        
        Args:
            symbol: IPA plosive symbol
            
        Returns:
            True if voiced, False otherwise
        """
        if symbol in CONSONANT_DATA:
            return CONSONANT_DATA[symbol].voiced
        return False
    
    def get_burst_frequency(self, symbol: str) -> float:
        """
        Get the burst center frequency for a plosive symbol.
        
        Args:
            symbol: IPA plosive symbol
            
        Returns:
            Center frequency in Hz, or 0 if not a plosive
        """
        if symbol in CONSONANT_DATA:
            return CONSONANT_DATA[symbol].noise_freq
        return 0.0
