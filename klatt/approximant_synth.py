"""
approximant_synth.py

Provides approximant consonant synthesis using the Klatt formant synthesizer.
Approximants are vowel-like consonants with formant structure but more
constricted articulation.

Classes:
    ApproximantSynthesizer: Synthesizes approximant consonants

Requirements:
    - 5.1: Synthesize approximants (ʋ, ɹ, ɻ, j, ɰ) using vowel-like formant parameters
    - 5.2: Use tdklatt.py's formant synthesis method
    - 5.3: Maintain voicing (AV > 0) for all approximants (all are voiced)
    - 12.3: Use tdklatt.py for approximant and lateral synthesis
"""

import numpy as np
from scipy.signal import resample_poly
from typing import Optional, Dict
from math import gcd

from klatt.consonant_data import ConsonantParams, CONSONANT_DATA, APPROXIMANTS
from klatt.tdklatt import KlattParam1980, klatt_make


class ApproximantSynthesizer:
    """
    Synthesizes approximant consonants using the Klatt formant synthesizer.
    
    Approximants are produced with:
    - Vowel-like formant structure (F1-F3)
    - Voicing: All approximants are voiced (AV > 0)
    - More constricted articulation than vowels but no turbulence
    
    The formant values distinguish different approximants:
    - /j/ (palatal): High F2 (~2200 Hz), similar to /i/
    - /w/ (labiovelar): Low F2 (~700 Hz), similar to /u/
    - /ɹ/ (alveolar): Low F3 (~1700 Hz), characteristic of English /r/
    - /ʋ/ (labiodental): Intermediate formants
    - /ɰ/ (velar): High F2, back vowel-like
    
    Attributes:
        fs: Target sampling frequency in Hz (output will be resampled)
        f0: Default fundamental frequency for voicing
        klatt_fs: Internal Klatt synthesizer sampling rate (10000 Hz)
    """
    
    # Klatt synthesizer internal sample rate
    KLATT_FS = 10000
    
    # Default approximant parameters
    DEFAULT_AV = 60      # Voicing amplitude (dB)
    DEFAULT_F0 = 120.0   # Default fundamental frequency (Hz)
    
    def __init__(self, fs: int = 16000, f0: float = 120.0):
        """
        Initialize the approximant synthesizer.
        
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
        Synthesize an approximant consonant.
        
        Uses the Klatt synthesizer with formant parameters to create
        the vowel-like approximant sound.
        
        Args:
            params: ConsonantParams object containing acoustic parameters
            duration_ms: Duration of the approximant in milliseconds
            context: Optional context information (preceding/following segments)
            
        Returns:
            Audio waveform as numpy array (normalized to [-1, 1])
        """
        if duration_ms <= 0:
            return np.array([])
        
        # Create Klatt parameters for approximant synthesis
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
        Create Klatt synthesizer parameters for approximant synthesis.
        
        Sets up:
        - Formants F1-F3 based on place of articulation
        - Voicing amplitude (AV > 0 for all approximants)
        - No frication noise (AF = 0)
        
        Args:
            params: ConsonantParams with approximant acoustic parameters
            duration_ms: Duration in milliseconds
            
        Returns:
            KlattParam1980 object configured for approximant synthesis
        """
        duration_sec = duration_ms / 1000.0
        
        # Create Klatt parameters
        klatt_params = KlattParam1980(
            FS=self.klatt_fs,
            DUR=duration_sec,
            F0=self.f0,
            # Voicing - all approximants are voiced (Requirement 5.3)
            AV=self.DEFAULT_AV,
            AVS=0,
            AH=0,
            AF=0,  # No frication noise for approximants
            # No nasal pole/zero for approximants
            FNP=250,
            BNP=100,
            FNZ=250,
            BNZ=100,
            # Use cascade synthesis (SW=0)
            SW=0,
            # Formants based on place of articulation (Requirement 5.1)
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
        Get the formant frequencies for an approximant symbol.
        
        Args:
            symbol: IPA approximant symbol
            
        Returns:
            Tuple of (F1, F2, F3) in Hz, or (0, 0, 0) if not an approximant
        """
        if symbol in CONSONANT_DATA:
            params = CONSONANT_DATA[symbol]
            return (params.f1, params.f2, params.f3)
        return (0.0, 0.0, 0.0)
    
    def is_approximant(self, symbol: str) -> bool:
        """
        Check if a symbol is an approximant consonant.
        
        Args:
            symbol: IPA symbol
            
        Returns:
            True if approximant, False otherwise
        """
        return symbol in APPROXIMANTS
    
    def is_voiced(self, symbol: str) -> bool:
        """
        Check if an approximant symbol is voiced.
        
        All approximants are voiced by definition.
        
        Args:
            symbol: IPA approximant symbol
            
        Returns:
            True if the symbol is an approximant (all approximants are voiced)
        """
        if symbol in CONSONANT_DATA:
            return CONSONANT_DATA[symbol].voiced
        return False
