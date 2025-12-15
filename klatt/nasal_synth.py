"""
nasal_synth.py

Provides nasal consonant synthesis using the Klatt synthesizer with
nasal pole (FNP) and nasal zero (FNZ) anti-formant parameters.

Classes:
    NasalSynthesizer: Synthesizes nasal consonants

Requirements:
    - 2.1: Synthesize nasals using formant and anti-formant parameters
    - 2.2: Set nasal pole frequency (FNP) and bandwidth (BNP) by place
    - 2.3: Set nasal zero frequency (FNZ) and bandwidth (BNZ) for nasal cavity
    - 2.4: Maintain voicing (AV > 0) for all nasals (all nasals are voiced)
    - 12.4: Extend tdklatt.py to support anti-formants (already supported)
"""

import numpy as np
from scipy.signal import resample_poly
from typing import Optional, Dict

from klatt.consonant_data import ConsonantParams, CONSONANT_DATA, NASALS
from klatt.tdklatt import KlattParam1980, klatt_make


class NasalSynthesizer:
    """
    Synthesizes nasal consonants using the Klatt formant synthesizer.
    
    Nasals are produced with:
    - Nasal pole (FNP): Low-frequency resonance from nasal cavity (~250 Hz)
    - Nasal zero (FNZ): Anti-formant that attenuates oral cavity resonances
    - Voicing: All nasals are voiced (AV > 0)
    - Formants: F1-F3 vary by place of articulation
    
    The nasal zero (anti-formant) is the key acoustic feature that
    distinguishes nasals from other voiced consonants. It creates a
    spectral "notch" that attenuates energy at certain frequencies.
    
    Attributes:
        fs: Target sampling frequency in Hz (output will be resampled)
        f0: Default fundamental frequency for voicing
        klatt_fs: Internal Klatt synthesizer sampling rate (10000 Hz)
    """
    
    # Klatt synthesizer internal sample rate
    KLATT_FS = 10000
    
    # Default nasal parameters
    DEFAULT_FNP = 250    # Nasal pole frequency (Hz)
    DEFAULT_BNP = 100    # Nasal pole bandwidth (Hz)
    DEFAULT_BNZ = 100    # Nasal zero bandwidth (Hz)
    DEFAULT_AV = 60      # Voicing amplitude (dB)
    
    def __init__(self, fs: int = 16000, f0: float = 120.0):
        """
        Initialize the nasal synthesizer.
        
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
        Synthesize a nasal consonant.
        
        Uses the Klatt synthesizer with nasal pole and nasal zero parameters
        to create the characteristic nasal sound.
        
        Args:
            params: ConsonantParams object containing acoustic parameters
            duration_ms: Duration of the nasal in milliseconds
            context: Optional context information (preceding/following segments)
            
        Returns:
            Audio waveform as numpy array (normalized to [-1, 1])
        """
        if duration_ms <= 0:
            return np.array([])
        
        # Create Klatt parameters for nasal synthesis
        klatt_params = self._create_klatt_params(params, duration_ms)
        
        # Run the Klatt synthesizer
        synth = klatt_make(klatt_params)
        synth.run()
        
        # Resample to target sample rate
        output = self._resample_output(synth.output)
        
        return self._normalize(output)
    
    def _create_klatt_params(self, params: ConsonantParams, 
                              duration_ms: float) -> KlattParam1980:
        """
        Create Klatt synthesizer parameters for nasal synthesis.
        
        Sets up:
        - Nasal pole (FNP) and bandwidth (BNP)
        - Nasal zero (FNZ) and bandwidth (BNZ) 
        - Formants F1-F3 based on place of articulation
        - Voicing amplitude (AV > 0 for all nasals)
        
        Args:
            params: ConsonantParams with nasal acoustic parameters
            duration_ms: Duration in milliseconds
            
        Returns:
            KlattParam1980 object configured for nasal synthesis
        """
        duration_sec = duration_ms / 1000.0
        
        # Get nasal-specific parameters from ConsonantParams
        fnp = params.fnp if params.fnp > 0 else self.DEFAULT_FNP
        fnz = params.fnz if params.fnz > 0 else params.f2  # FNZ often near F2
        
        # Create Klatt parameters
        klatt_params = KlattParam1980(
            FS=self.klatt_fs,
            DUR=duration_sec,
            F0=self.f0,
            # Voicing - all nasals are voiced (Requirement 2.4)
            AV=self.DEFAULT_AV,
            AVS=0,
            AH=0,
            AF=0,
            # Nasal pole and zero (Requirements 2.2, 2.3)
            FNP=fnp,
            BNP=self.DEFAULT_BNP,
            FNZ=fnz,
            BNZ=self.DEFAULT_BNZ,
            # Use cascade synthesis (SW=0)
            SW=0,
            # Formants based on place of articulation
            FF=[params.f1, params.f2, params.f3, 3500, 4500],
            BW=[80, 100, 120, 200, 250],
        )
        
        return klatt_params
    
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
        
        # Calculate resampling ratio
        # From 10000 Hz to target fs
        if self.fs == self.klatt_fs:
            return audio
        
        # Use rational resampling
        # Find GCD for efficient resampling
        from math import gcd
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
    
    def get_fnp(self, symbol: str) -> float:
        """
        Get the nasal pole frequency for a nasal symbol.
        
        Args:
            symbol: IPA nasal symbol
            
        Returns:
            Nasal pole frequency in Hz, or 0 if not a nasal
        """
        if symbol in CONSONANT_DATA:
            return CONSONANT_DATA[symbol].fnp
        return 0.0
    
    def get_fnz(self, symbol: str) -> float:
        """
        Get the nasal zero (anti-formant) frequency for a nasal symbol.
        
        Args:
            symbol: IPA nasal symbol
            
        Returns:
            Nasal zero frequency in Hz, or 0 if not a nasal
        """
        if symbol in CONSONANT_DATA:
            return CONSONANT_DATA[symbol].fnz
        return 0.0
    
    def is_nasal(self, symbol: str) -> bool:
        """
        Check if a symbol is a nasal consonant.
        
        Args:
            symbol: IPA symbol
            
        Returns:
            True if nasal, False otherwise
        """
        return symbol in NASALS
    
    def is_voiced(self, symbol: str) -> bool:
        """
        Check if a nasal symbol is voiced.
        
        All nasals are voiced by definition.
        
        Args:
            symbol: IPA nasal symbol
            
        Returns:
            True if the symbol is a nasal (all nasals are voiced)
        """
        if symbol in CONSONANT_DATA:
            return CONSONANT_DATA[symbol].voiced
        return False
