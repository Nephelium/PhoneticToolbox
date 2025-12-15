"""
tap_trill_synth.py

Provides tap/flap and trill consonant synthesis.
Taps are single quick contacts with very short duration (20-40ms).
Trills are multiple vibrations with periodic amplitude modulation.

Classes:
    TapTrillSynthesizer: Synthesizes tap and trill consonants

Requirements:
    - 6.1: Synthesize taps (ⱱ, ɾ, ɽ) as single quick contact
    - 6.2: Synthesize trills (ʙ, r, ʀ) as multiple vibrations
    - 6.3: Tap duration is very short (~20-40ms)
    - 6.4: Trills have periodic amplitude modulation
"""

import numpy as np
from scipy.signal import resample_poly
from typing import Optional, Dict
from math import gcd

from klatt.consonant_data import ConsonantParams, CONSONANT_DATA, TAPS, TRILLS
from klatt.tdklatt import KlattParam1980, klatt_make


class TapTrillSynthesizer:
    """
    Synthesizes tap/flap and trill consonants.
    
    Taps (ⱱ, ɾ, ɽ):
    - Single quick contact between articulators
    - Very short duration (20-40ms) - Requirement 6.3
    - Voiced, vowel-like formant structure
    - Rapid attack and decay envelope
    
    Trills (ʙ, r, ʀ):
    - Multiple vibrations of an articulator
    - Periodic amplitude modulation - Requirement 6.4
    - Voiced throughout
    - Typical trill rate: 20-30 Hz (cycles per second)
    
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
    
    # Tap duration constraints (Requirement 6.3)
    TAP_MIN_DURATION_MS = 20.0
    TAP_MAX_DURATION_MS = 40.0
    TAP_DEFAULT_DURATION_MS = 30.0
    
    # Trill parameters (Requirement 6.4)
    TRILL_RATE_HZ = 25.0  # Typical trill vibration rate (20-30 Hz)
    TRILL_MODULATION_DEPTH = 0.6  # Amplitude modulation depth (0-1)
    
    def __init__(self, fs: int = 16000, f0: float = 120.0):
        """
        Initialize the tap/trill synthesizer.
        
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
        Synthesize a tap or trill consonant.
        
        Routes to appropriate synthesis method based on manner of articulation.
        
        Args:
            params: ConsonantParams object containing acoustic parameters
            duration_ms: Duration in milliseconds (constrained for taps)
            context: Optional context information (preceding/following segments)
            
        Returns:
            Audio waveform as numpy array (normalized to [-1, 1])
        """
        if duration_ms <= 0:
            return np.array([])
        
        manner = params.manner
        
        if manner == 'tap':
            return self._synthesize_tap(params, duration_ms, context)
        elif manner == 'trill':
            return self._synthesize_trill(params, duration_ms, context)
        else:
            # Default to tap-like synthesis for unknown manner
            return self._synthesize_tap(params, duration_ms, context)


    def _synthesize_tap(self, params: ConsonantParams, duration_ms: float,
                        context: Optional[Dict] = None) -> np.ndarray:
        """
        Synthesize a tap/flap consonant (ⱱ, ɾ, ɽ).
        
        Taps are characterized by:
        - Single quick contact between articulators
        - Very short duration (20-40ms) - Requirement 6.3
        - Voiced, vowel-like formant structure
        - Rapid attack and decay envelope
        
        Requirements: 6.1, 6.3
        
        Args:
            params: ConsonantParams with tap acoustic parameters
            duration_ms: Requested duration (constrained to 20-40ms)
            context: Optional context information
            
        Returns:
            Audio waveform as numpy array
        """
        # Constrain tap duration to valid range (Requirement 6.3)
        actual_duration = self._constrain_tap_duration(duration_ms, params)
        
        # Create Klatt parameters for tap synthesis
        klatt_params = self._create_klatt_params(params, actual_duration)
        
        # Run the Klatt synthesizer
        synth = klatt_make(klatt_params)
        synth.run()
        
        # Resample to target sample rate
        output = self._resample_output(synth.output)
        
        # Apply rapid attack/decay envelope characteristic of taps
        output = self._apply_tap_envelope(output)
        
        return self._normalize(output)
    
    def _synthesize_trill(self, params: ConsonantParams, duration_ms: float,
                          context: Optional[Dict] = None) -> np.ndarray:
        """
        Synthesize a trill consonant (ʙ, r, ʀ).
        
        Trills are characterized by:
        - Multiple vibrations of an articulator
        - Periodic amplitude modulation - Requirement 6.4
        - Voiced throughout
        - Typical trill rate: 20-30 Hz
        
        Requirements: 6.2, 6.4
        
        Args:
            params: ConsonantParams with trill acoustic parameters
            duration_ms: Duration in milliseconds
            context: Optional context information
            
        Returns:
            Audio waveform as numpy array
        """
        # Use the default duration from params (trills have fixed duration)
        actual_duration = params.default_duration
        
        # Create Klatt parameters for trill synthesis
        klatt_params = self._create_klatt_params(params, actual_duration)
        
        # Run the Klatt synthesizer
        synth = klatt_make(klatt_params)
        synth.run()
        
        # Resample to target sample rate
        output = self._resample_output(synth.output)
        
        # Apply periodic amplitude modulation (Requirement 6.4)
        output = self._apply_trill_modulation(output, params)
        
        return self._normalize(output)
    
    def _constrain_tap_duration(self, duration_ms: float, 
                                 params: ConsonantParams) -> float:
        """
        Constrain tap duration to valid range.
        
        Requirement 6.3: Tap duration is very short (~20-40ms).
        Taps have fixed duration and cannot be adjusted by modifiers.
        
        Args:
            duration_ms: Requested duration
            params: ConsonantParams for the tap
            
        Returns:
            Constrained duration within [TAP_MIN_DURATION_MS, TAP_MAX_DURATION_MS]
        """
        # Use the default duration from params, but constrain to valid range
        target_duration = params.default_duration
        return max(self.TAP_MIN_DURATION_MS, 
                   min(target_duration, self.TAP_MAX_DURATION_MS))
    
    def _create_klatt_params(self, params: ConsonantParams, 
                              duration_ms: float) -> KlattParam1980:
        """
        Create Klatt synthesizer parameters for tap/trill synthesis.
        
        Both taps and trills use vowel-like formant synthesis with voicing.
        
        Args:
            params: ConsonantParams with acoustic parameters
            duration_ms: Duration in milliseconds
            
        Returns:
            KlattParam1980 object configured for synthesis
        """
        duration_sec = duration_ms / 1000.0
        
        # Create Klatt parameters
        klatt_params = KlattParam1980(
            FS=self.klatt_fs,
            DUR=duration_sec,
            F0=self.f0,
            # Voicing - all taps and trills are voiced
            AV=self.DEFAULT_AV,
            AVS=0,
            AH=0,
            AF=0,
            # No nasal pole/zero
            FNP=250,
            BNP=100,
            FNZ=250,
            BNZ=100,
            # Use cascade synthesis (SW=0)
            SW=0,
            # Formants based on place of articulation
            FF=[params.f1, params.f2, params.f3, 3500, 4500],
            BW=[80, 100, 120, 200, 250],
        )
        
        return klatt_params
    
    def _apply_tap_envelope(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply rapid attack/decay envelope for tap sounds.
        
        Taps have very rapid onset and offset, creating a brief
        "tap" or "flap" sound characteristic.
        
        Args:
            audio: Input audio array
            
        Returns:
            Audio with tap envelope applied
        """
        if len(audio) == 0:
            return audio
        
        n_samples = len(audio)
        
        # Very rapid attack (5% of duration)
        attack_samples = max(1, int(n_samples * 0.05))
        # Rapid decay (10% of duration)
        decay_samples = max(1, int(n_samples * 0.10))
        
        envelope = np.ones(n_samples)
        
        # Attack phase - rapid rise
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase - rapid fall
        if decay_samples > 0:
            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        return audio * envelope


    def _apply_trill_modulation(self, audio: np.ndarray, 
                                 params: ConsonantParams) -> np.ndarray:
        """
        Apply periodic amplitude modulation for trill sounds.
        
        Requirement 6.4: Trills have periodic amplitude modulation.
        
        The modulation simulates the periodic opening and closing
        of the vocal tract during trill production. Typical trill
        rates are 20-30 Hz.
        
        Args:
            audio: Input audio array
            params: ConsonantParams for the trill
            
        Returns:
            Audio with trill modulation applied
        """
        if len(audio) == 0:
            return audio
        
        n_samples = len(audio)
        
        # Get trill rate based on place of articulation
        trill_rate = self._get_trill_rate(params)
        
        # Create time array
        t = np.arange(n_samples) / self.fs
        
        # Create amplitude modulation envelope
        # Using a raised cosine for smooth modulation
        # Modulation oscillates between (1 - depth) and 1
        modulation = 1.0 - self.TRILL_MODULATION_DEPTH * 0.5 * (1 - np.cos(2 * np.pi * trill_rate * t))
        
        # Apply overall envelope (gentle attack and decay)
        envelope = self._create_trill_envelope(n_samples)
        
        return audio * modulation * envelope
    
    def _get_trill_rate(self, params: ConsonantParams) -> float:
        """
        Get the trill vibration rate based on place of articulation.
        
        Different places of articulation have slightly different
        typical trill rates:
        - Bilabial (ʙ): ~25 Hz
        - Alveolar (r): ~25 Hz
        - Uvular (ʀ): ~20 Hz (slightly slower due to larger mass)
        
        Args:
            params: ConsonantParams for the trill
            
        Returns:
            Trill rate in Hz
        """
        place = params.place
        
        if place == 'uvular':
            return 20.0  # Uvular trills are typically slower
        elif place == 'bilabial':
            return 25.0
        else:
            return self.TRILL_RATE_HZ  # Default alveolar rate
    
    def _create_trill_envelope(self, n_samples: int) -> np.ndarray:
        """
        Create overall envelope for trill sounds.
        
        Trills have a gentle attack and decay to sound natural.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Envelope array
        """
        if n_samples <= 0:
            return np.array([])
        
        # Gentle attack (10% of duration)
        attack_samples = max(1, int(n_samples * 0.10))
        # Gentle decay (15% of duration)
        decay_samples = max(1, int(n_samples * 0.15))
        
        envelope = np.ones(n_samples)
        
        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        if decay_samples > 0:
            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        return envelope
    
    def _resample_output(self, audio: np.ndarray) -> np.ndarray:
        """
        Resample Klatt output to target sample rate.
        
        The Klatt synthesizer runs at 10000 Hz internally.
        This method resamples to the target sample rate.
        
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
        Get the formant frequencies for a tap/trill symbol.
        
        Args:
            symbol: IPA tap/trill symbol
            
        Returns:
            Tuple of (F1, F2, F3) in Hz, or (0, 0, 0) if not found
        """
        if symbol in CONSONANT_DATA:
            params = CONSONANT_DATA[symbol]
            return (params.f1, params.f2, params.f3)
        return (0.0, 0.0, 0.0)
    
    def is_tap(self, symbol: str) -> bool:
        """
        Check if a symbol is a tap/flap consonant.
        
        Args:
            symbol: IPA symbol
            
        Returns:
            True if tap/flap, False otherwise
        """
        return symbol in TAPS
    
    def is_trill(self, symbol: str) -> bool:
        """
        Check if a symbol is a trill consonant.
        
        Args:
            symbol: IPA symbol
            
        Returns:
            True if trill, False otherwise
        """
        return symbol in TRILLS
    
    def is_voiced(self, symbol: str) -> bool:
        """
        Check if a tap/trill symbol is voiced.
        
        All taps and trills are voiced by definition.
        
        Args:
            symbol: IPA tap/trill symbol
            
        Returns:
            True if the symbol is a tap/trill (all are voiced)
        """
        if symbol in CONSONANT_DATA:
            return CONSONANT_DATA[symbol].voiced
        return False
    
    def get_tap_duration_range(self) -> tuple:
        """
        Get the valid duration range for taps.
        
        Returns:
            Tuple of (min_duration_ms, max_duration_ms)
        """
        return (self.TAP_MIN_DURATION_MS, self.TAP_MAX_DURATION_MS)
    
    def get_trill_rate(self, symbol: str) -> float:
        """
        Get the trill vibration rate for a trill symbol.
        
        Args:
            symbol: IPA trill symbol
            
        Returns:
            Trill rate in Hz, or 0 if not a trill
        """
        if symbol in CONSONANT_DATA and CONSONANT_DATA[symbol].manner == 'trill':
            return self._get_trill_rate(CONSONANT_DATA[symbol])
        return 0.0
