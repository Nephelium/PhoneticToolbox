"""
transition_gen.py

Provides transition (音渡段) generation between consonants and vowels.
Transitions create smooth acoustic connections between adjacent segments.

Classes:
    TransitionGenerator: Generates transition segments between consonants and vowels

Requirements:
    - 10.1: Insert transition between consonant and following vowel
    - 10.2: Insert transition between vowel and following consonant
    - 10.3: Use formant smoothing for smoothable consonants
    - 10.4: Use short approximant for non-smoothable consonants
    - 10.5: Transition duration in range 20-50ms
    - 2.5, 2.6: Nasal-vowel smooth formant transitions
    - 3.5: Plosive-vowel burst + formant transition
    - 4.7: Fricative-vowel noise to periodic transition
    - 5.4: Approximant-vowel smooth formant transitions
"""

import numpy as np
from scipy.signal import resample_poly
from typing import Optional, Dict, Tuple, List
from math import gcd

try:
    from .consonant_data import (
        CONSONANT_DATA, ConsonantParams,
        NASALS, PLOSIVES, SIBILANTS, FRICATIVES,
        APPROXIMANTS, LATERAL_APPROXIMANTS, TAPS, TRILLS
    )
    from .tdklatt import KlattParam1980, klatt_make
except ImportError:
    from consonant_data import (
        CONSONANT_DATA, ConsonantParams,
        NASALS, PLOSIVES, SIBILANTS, FRICATIVES,
        APPROXIMANTS, LATERAL_APPROXIMANTS, TAPS, TRILLS
    )
    from tdklatt import KlattParam1980, klatt_make


# Default vowel formants for reference
DEFAULT_VOWEL_FORMANTS = {
    'i': [270, 2290, 3010], 'y': [270, 1800, 2200],
    'ɨ': [290, 1400, 2100], 'ʉ': [290, 1200, 2100],
    'ɯ': [300, 1100, 2200], 'u': [300, 870, 2240],
    'ɪ': [390, 1990, 2550], 'ʏ': [390, 1500, 2100],
    'ʊ': [440, 1020, 2240],
    'e': [390, 2030, 2600], 'ø': [390, 1550, 2200],
    'ɘ': [390, 1300, 2200], 'ɵ': [390, 1100, 2200],
    'ɤ': [460, 1100, 2300], 'o': [460, 800, 2250],
    'ə': [500, 1500, 2500],
    'ɛ': [530, 1840, 2480], 'œ': [530, 1300, 2200],
    'ɜ': [560, 1350, 2200], 'ɞ': [560, 1100, 2200],
    'ʌ': [640, 1200, 2400], 'ɔ': [570, 840, 2410],
    'æ': [660, 1720, 2410], 'ɐ': [660, 1400, 2300],
    'a': [730, 1090, 2440], 'ɶ': [730, 1000, 2200],
    'ɑ': [730, 1100, 2400], 'ɒ': [730, 850, 2300],
}

# Schwa formants as default neutral vowel
SCHWA_FORMANTS = [500, 1500, 2500]

# Consonant types that can use formant smoothing
SMOOTHABLE_MANNERS = {'nasal', 'approximant', 'lateral_approximant'}

# Mapping from place of articulation to nearby approximant for transitions
PLACE_TO_APPROXIMANT = {
    'bilabial': 'w',
    'labiodental': 'ʋ',
    'dental': 'ɹ',
    'alveolar': 'ɹ',
    'postalveolar': 'ɹ',
    'retroflex': 'ɻ',
    'alveolopalatal': 'j',
    'palatal': 'j',
    'velar': 'ɰ',
    'labiovelar': 'w',
    'uvular': 'ʁ',
    'pharyngeal': 'ʕ',
    'epiglottal': 'ʕ',
    'glottal': 'h',
}


class TransitionGenerator:
    """
    Generates acoustic transitions between consonants and vowels.
    
    Transitions are short segments (20-50ms) that create smooth
    acoustic connections between adjacent segments. The type of
    transition depends on the consonant manner:
    
    - Smoothable (nasal, approximant, lateral_approximant):
      Use formant interpolation for smooth transition
    - Non-smoothable (plosive, fricative, tap, trill):
      Use short approximant as transition bridge
    
    Attributes:
        fs: Target sampling frequency in Hz
        f0: Default fundamental frequency for voicing
        default_duration_ms: Default transition duration (30ms)
        min_duration_ms: Minimum transition duration (20ms)
        max_duration_ms: Maximum transition duration (50ms)
        vowel_formants: Dictionary of vowel symbols to formant values
    """
    
    # Klatt synthesizer internal sample rate
    KLATT_FS = 10000
    
    # Default parameters
    DEFAULT_DURATION_MS = 30.0
    MIN_DURATION_MS = 20.0
    MAX_DURATION_MS = 50.0
    DEFAULT_AV = 60
    DEFAULT_F0 = 120.0
    
    def __init__(self, fs: int = 16000, f0: float = 120.0,
                 vowel_formants: Optional[Dict] = None):
        """
        Initialize the transition generator.
        
        Args:
            fs: Target output sampling frequency in Hz (default 16000)
            f0: Default fundamental frequency for voicing (default 120 Hz)
            vowel_formants: Dictionary of vowel symbols to formant values
        """
        self.fs = fs
        self.f0 = f0
        self.klatt_fs = self.KLATT_FS
        self.default_duration_ms = self.DEFAULT_DURATION_MS
        self.min_duration_ms = self.MIN_DURATION_MS
        self.max_duration_ms = self.MAX_DURATION_MS
        self.vowel_formants = vowel_formants or DEFAULT_VOWEL_FORMANTS
    
    def generate(self, from_segment: Dict, to_segment: Dict,
                 duration_ms: Optional[float] = None) -> np.ndarray:
        """
        Generate a transition between two segments.
        
        Args:
            from_segment: Source segment info with keys:
                - type: 'vowel', 'consonant', or 'silence'
                - symbol: IPA symbol
                - f1, f2, f3: Optional formant values
            to_segment: Target segment info (same structure)
            duration_ms: Optional transition duration (default 30ms)
            
        Returns:
            Audio waveform as numpy array (normalized to [-1, 1])
        """
        if duration_ms is None:
            duration_ms = self.default_duration_ms
        
        # Clamp duration to valid range
        duration_ms = max(self.min_duration_ms, 
                         min(self.max_duration_ms, duration_ms))
        
        if duration_ms <= 0:
            return np.array([])
        
        from_type = from_segment.get('type', 'silence')
        to_type = to_segment.get('type', 'silence')
        
        # Determine if we can use formant smoothing
        if self._can_smooth(from_segment, to_segment):
            return self._formant_transition(from_segment, to_segment, duration_ms)
        else:
            return self._approximant_transition(from_segment, to_segment, duration_ms)
    
    def can_smooth(self, from_segment: Dict, to_segment: Dict) -> bool:
        """
        Public method to check if formant smoothing can be used.
        
        Args:
            from_segment: Source segment info
            to_segment: Target segment info
            
        Returns:
            True if formant smoothing can be used
        """
        return self._can_smooth(from_segment, to_segment)
    
    def _can_smooth(self, from_segment: Dict, to_segment: Dict) -> bool:
        """
        Determine if formant smoothing can be used for this transition.
        
        Formant smoothing works for:
        - vowel <-> vowel
        - vowel <-> smoothable consonant (nasal, approximant, lateral_approximant)
        - smoothable consonant <-> smoothable consonant
        
        Args:
            from_segment: Source segment info
            to_segment: Target segment info
            
        Returns:
            True if formant smoothing can be used
        """
        from_type = from_segment.get('type', 'silence')
        to_type = to_segment.get('type', 'silence')
        
        # Check if from_segment is smoothable
        from_smoothable = self._is_smoothable(from_segment)
        
        # Check if to_segment is smoothable
        to_smoothable = self._is_smoothable(to_segment)
        
        return from_smoothable and to_smoothable
    
    def _is_smoothable(self, segment: Dict) -> bool:
        """
        Check if a segment can participate in formant smoothing.
        
        Args:
            segment: Segment info dictionary
            
        Returns:
            True if the segment is smoothable
        """
        seg_type = segment.get('type', 'silence')
        
        if seg_type == 'vowel':
            return True
        
        if seg_type == 'consonant':
            symbol = segment.get('symbol', '')
            if symbol in CONSONANT_DATA:
                manner = CONSONANT_DATA[symbol].manner
                return manner in SMOOTHABLE_MANNERS
        
        return False
    
    def _get_formants(self, segment: Dict) -> Tuple[float, float, float]:
        """
        Get formant values for a segment.
        
        Args:
            segment: Segment info dictionary
            
        Returns:
            Tuple of (F1, F2, F3) in Hz
        """
        # Check if formants are explicitly provided
        if 'f1' in segment and 'f2' in segment and 'f3' in segment:
            return (segment['f1'], segment['f2'], segment['f3'])
        
        seg_type = segment.get('type', 'silence')
        symbol = segment.get('symbol', '')
        
        if seg_type == 'vowel':
            formants = self.vowel_formants.get(symbol, SCHWA_FORMANTS)
            return (formants[0], formants[1], formants[2])
        
        if seg_type == 'consonant' and symbol in CONSONANT_DATA:
            params = CONSONANT_DATA[symbol]
            if params.f1 > 0 and params.f2 > 0 and params.f3 > 0:
                return (params.f1, params.f2, params.f3)
        
        # Default to schwa formants
        return (SCHWA_FORMANTS[0], SCHWA_FORMANTS[1], SCHWA_FORMANTS[2])
    
    def _formant_transition(self, from_segment: Dict, to_segment: Dict,
                            duration_ms: float) -> np.ndarray:
        """
        Generate a smooth formant interpolation transition.
        
        Uses linear interpolation of formant frequencies to create
        a smooth acoustic transition between segments.
        
        Args:
            from_segment: Source segment info
            to_segment: Target segment info
            duration_ms: Transition duration in milliseconds
            
        Returns:
            Audio waveform as numpy array
        """
        duration_sec = duration_ms / 1000.0
        
        # Create Klatt parameters first to get the exact sample count
        klatt_params = KlattParam1980(
            FS=self.klatt_fs,
            DUR=duration_sec,
            F0=self.f0,
            AV=self.DEFAULT_AV,
            AVS=0,
            AH=0,
            AF=0,
            FNP=250,
            BNP=100,
            FNZ=250,
            BNZ=100,
            SW=0,
            FF=[500, 1500, 2500, 3500, 4500],
            BW=[80, 100, 120, 200, 250],
        )
        
        # Use the exact sample count from KlattParam1980
        n_samples_klatt = klatt_params.N_SAMP
        
        if n_samples_klatt <= 0:
            return np.array([])
        
        # Get formants for both segments
        f1_start, f2_start, f3_start = self._get_formants(from_segment)
        f1_end, f2_end, f3_end = self._get_formants(to_segment)
        
        # Create time-varying formant arrays with exact sample count
        t = np.linspace(0, 1, n_samples_klatt)
        f1 = f1_start + (f1_end - f1_start) * t
        f2 = f2_start + (f2_end - f2_start) * t
        f3 = f3_start + (f3_end - f3_start) * t
        
        # Set time-varying formants
        klatt_params.FF[0][:] = f1
        klatt_params.FF[1][:] = f2
        klatt_params.FF[2][:] = f3
        
        # Run synthesizer
        synth = klatt_make(klatt_params)
        synth.run()
        
        # Resample to target sample rate
        output = self._resample_output(synth.output)
        
        return self._normalize(output)
    
    def _approximant_transition(self, from_segment: Dict, to_segment: Dict,
                                duration_ms: float) -> np.ndarray:
        """
        Generate a transition using a short approximant.
        
        For non-smoothable consonants (plosives, fricatives, etc.),
        uses a short approximant sound as a transition bridge.
        
        Args:
            from_segment: Source segment info
            to_segment: Target segment info
            duration_ms: Transition duration in milliseconds
            
        Returns:
            Audio waveform as numpy array
        """
        # Check for specific consonant types that need special handling
        from_type = from_segment.get('type', 'silence')
        to_type = to_segment.get('type', 'silence')
        
        # Determine consonant segment and vowel segment
        consonant_seg = None
        vowel_seg = None
        is_cv = False  # consonant-vowel order
        
        if from_type == 'consonant' and to_type == 'vowel':
            consonant_seg = from_segment
            vowel_seg = to_segment
            is_cv = True
        elif from_type == 'vowel' and to_type == 'consonant':
            consonant_seg = to_segment
            vowel_seg = from_segment
            is_cv = False
        
        if consonant_seg:
            symbol = consonant_seg.get('symbol', '')
            if symbol in CONSONANT_DATA:
                manner = CONSONANT_DATA[symbol].manner
                
                # Use specific transition methods based on manner
                if manner == 'plosive':
                    return self._plosive_vowel_transition(
                        consonant_seg, vowel_seg, duration_ms, is_cv)
                elif manner in ('sibilant', 'fricative', 'lateral_fricative'):
                    return self._fricative_vowel_transition(
                        consonant_seg, vowel_seg, duration_ms, is_cv)
        
        # Default: use short approximant
        return self._default_approximant_transition(from_segment, to_segment, duration_ms)
    
    def _default_approximant_transition(self, from_segment: Dict, to_segment: Dict,
                                        duration_ms: float) -> np.ndarray:
        """
        Generate a default transition using a short approximant.
        
        Args:
            from_segment: Source segment info
            to_segment: Target segment info
            duration_ms: Transition duration in milliseconds
            
        Returns:
            Audio waveform as numpy array
        """
        # Determine which approximant to use based on place of articulation
        approx_symbol = self._select_approximant(from_segment, to_segment)
        
        # Get approximant formants
        if approx_symbol in CONSONANT_DATA:
            params = CONSONANT_DATA[approx_symbol]
            f1, f2, f3 = params.f1, params.f2, params.f3
        else:
            # Default to schwa-like formants
            f1, f2, f3 = SCHWA_FORMANTS
        
        duration_sec = duration_ms / 1000.0
        
        # Create Klatt parameters for approximant
        klatt_params = KlattParam1980(
            FS=self.klatt_fs,
            DUR=duration_sec,
            F0=self.f0,
            AV=self.DEFAULT_AV,
            AVS=0,
            AH=0,
            AF=0,
            FNP=250,
            BNP=100,
            FNZ=250,
            BNZ=100,
            SW=0,
            FF=[f1, f2, f3, 3500, 4500],
            BW=[80, 100, 120, 200, 250],
        )
        
        # Run synthesizer
        synth = klatt_make(klatt_params)
        synth.run()
        
        # Resample to target sample rate
        output = self._resample_output(synth.output)
        
        return self._normalize(output)
    
    def _plosive_vowel_transition(self, plosive_seg: Dict, vowel_seg: Dict,
                                   duration_ms: float, is_cv: bool) -> np.ndarray:
        """
        Generate plosive-vowel transition with burst + formant transition.
        
        For plosive-vowel (CV) transitions:
        - Short burst noise at the beginning
        - Rapid formant transition to vowel
        
        For vowel-plosive (VC) transitions:
        - Formant transition from vowel
        - Closure preparation
        
        Requirements: 3.5
        
        Args:
            plosive_seg: Plosive consonant segment info
            vowel_seg: Vowel segment info
            duration_ms: Transition duration in milliseconds
            is_cv: True if consonant-vowel order, False if vowel-consonant
            
        Returns:
            Audio waveform as numpy array
        """
        symbol = plosive_seg.get('symbol', '')
        params = CONSONANT_DATA.get(symbol)
        
        if params is None:
            return self._default_approximant_transition(plosive_seg, vowel_seg, duration_ms)
        
        # Get vowel formants
        v_f1, v_f2, v_f3 = self._get_formants(vowel_seg)
        
        # Get burst frequency characteristics from plosive
        burst_freq = params.noise_freq if params.noise_freq > 0 else 2000
        
        duration_sec = duration_ms / 1000.0
        
        # Create Klatt parameters first to get exact sample count
        klatt_params = KlattParam1980(
            FS=self.klatt_fs,
            DUR=duration_sec,
            F0=self.f0,
            AV=self.DEFAULT_AV,
            AVS=0,
            AH=0,
            AF=0,
            FNP=250,
            BNP=100,
            FNZ=250,
            BNZ=100,
            SW=0,
            FF=[500, 1500, 2500, 3500, 4500],
            BW=[80, 100, 120, 200, 250],
        )
        
        n_samples_klatt = klatt_params.N_SAMP
        
        if n_samples_klatt <= 0:
            return np.array([])
        
        # Create time array with exact sample count
        t = np.linspace(0, 1, n_samples_klatt)
        
        if is_cv:
            # CV transition: burst at start, then formant transition
            # Formants start from neutral and move to vowel
            f1 = 400 + (v_f1 - 400) * t
            f2 = 1500 + (v_f2 - 1500) * t
            f3 = 2500 + (v_f3 - 2500) * t
            
            # Voicing amplitude ramps up
            av = self.DEFAULT_AV * t
            
            # Small amount of aspiration at the start (for burst effect)
            ah = 30 * np.exp(-10 * t)
        else:
            # VC transition: formant transition to closure
            f1 = v_f1 + (400 - v_f1) * t
            f2 = v_f2 + (1500 - v_f2) * t
            f3 = v_f3 + (2500 - v_f3) * t
            
            # Voicing amplitude ramps down
            av = self.DEFAULT_AV * (1 - t)
            ah = np.zeros(n_samples_klatt)
        
        # Set time-varying parameters
        klatt_params.FF[0][:] = f1
        klatt_params.FF[1][:] = f2
        klatt_params.FF[2][:] = f3
        klatt_params.AV[:] = av
        klatt_params.AH[:] = ah
        
        # Run synthesizer
        synth = klatt_make(klatt_params)
        synth.run()
        
        # Resample to target sample rate
        output = self._resample_output(synth.output)
        
        return self._normalize(output)
    
    def _fricative_vowel_transition(self, fricative_seg: Dict, vowel_seg: Dict,
                                     duration_ms: float, is_cv: bool) -> np.ndarray:
        """
        Generate fricative-vowel transition with noise to periodic transition.
        
        For fricative-vowel (CV) transitions:
        - Noise amplitude decreases
        - Voicing amplitude increases
        - Formants transition to vowel
        
        For vowel-fricative (VC) transitions:
        - Voicing amplitude decreases
        - Noise amplitude increases
        - Formants transition from vowel
        
        Requirements: 4.7
        
        Args:
            fricative_seg: Fricative consonant segment info
            vowel_seg: Vowel segment info
            duration_ms: Transition duration in milliseconds
            is_cv: True if consonant-vowel order, False if vowel-consonant
            
        Returns:
            Audio waveform as numpy array
        """
        symbol = fricative_seg.get('symbol', '')
        params = CONSONANT_DATA.get(symbol)
        
        if params is None:
            return self._default_approximant_transition(fricative_seg, vowel_seg, duration_ms)
        
        # Get vowel formants
        v_f1, v_f2, v_f3 = self._get_formants(vowel_seg)
        
        duration_sec = duration_ms / 1000.0
        
        # Create Klatt parameters first to get exact sample count
        klatt_params = KlattParam1980(
            FS=self.klatt_fs,
            DUR=duration_sec,
            F0=self.f0,
            AV=self.DEFAULT_AV,
            AVS=0,
            AH=0,
            AF=0,
            FNP=250,
            BNP=100,
            FNZ=250,
            BNZ=100,
            SW=0,
            FF=[500, 1500, 2500, 3500, 4500],
            BW=[80, 100, 120, 200, 250],
        )
        
        n_samples_klatt = klatt_params.N_SAMP
        
        if n_samples_klatt <= 0:
            return np.array([])
        
        # Create time array with exact sample count
        t = np.linspace(0, 1, n_samples_klatt)
        
        # Determine if fricative is voiced
        is_voiced = params.voiced
        
        if is_cv:
            # CV transition: noise decreases, voicing increases
            f1 = 400 + (v_f1 - 400) * t
            f2 = 1500 + (v_f2 - 1500) * t
            f3 = 2500 + (v_f3 - 2500) * t
            
            # Voicing ramps up
            av = self.DEFAULT_AV * t
            
            # Frication ramps down
            af = 50 * (1 - t)
        else:
            # VC transition: voicing decreases, noise increases
            f1 = v_f1 + (400 - v_f1) * t
            f2 = v_f2 + (1500 - v_f2) * t
            f3 = v_f3 + (2500 - v_f3) * t
            
            # Voicing ramps down
            av = self.DEFAULT_AV * (1 - t)
            
            # Frication ramps up
            af = 50 * t
        
        # Set time-varying parameters
        klatt_params.FF[0][:] = f1
        klatt_params.FF[1][:] = f2
        klatt_params.FF[2][:] = f3
        klatt_params.AV[:] = av
        klatt_params.AF[:] = af
        
        # Run synthesizer
        synth = klatt_make(klatt_params)
        synth.run()
        
        # Resample to target sample rate
        output = self._resample_output(synth.output)
        
        return self._normalize(output)
    
    def generate_nasal_vowel_transition(self, nasal_seg: Dict, vowel_seg: Dict,
                                         duration_ms: float, is_cv: bool) -> np.ndarray:
        """
        Generate nasal-vowel transition with smooth formant transition.
        
        Nasals use formant smoothing since they have well-defined formants.
        This is a public method for explicit nasal transitions.
        
        Requirements: 2.5, 2.6
        
        Args:
            nasal_seg: Nasal consonant segment info
            vowel_seg: Vowel segment info
            duration_ms: Transition duration in milliseconds
            is_cv: True if consonant-vowel order, False if vowel-consonant
            
        Returns:
            Audio waveform as numpy array
        """
        if is_cv:
            return self._formant_transition(nasal_seg, vowel_seg, duration_ms)
        else:
            return self._formant_transition(vowel_seg, nasal_seg, duration_ms)
    
    def _select_approximant(self, from_segment: Dict, to_segment: Dict) -> str:
        """
        Select an appropriate approximant for transition.
        
        Chooses an approximant based on the place of articulation
        of the consonant involved in the transition.
        
        Args:
            from_segment: Source segment info
            to_segment: Target segment info
            
        Returns:
            IPA symbol of the selected approximant
        """
        # Try to get place from consonant segment
        place = None
        
        for segment in [from_segment, to_segment]:
            if segment.get('type') == 'consonant':
                symbol = segment.get('symbol', '')
                if symbol in CONSONANT_DATA:
                    place = CONSONANT_DATA[symbol].place
                    break
        
        if place and place in PLACE_TO_APPROXIMANT:
            return PLACE_TO_APPROXIMANT[place]
        
        # Default to alveolar approximant
        return 'ɹ'
    
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
    
    def get_transition_duration_ms(self) -> float:
        """
        Get the default transition duration.
        
        Returns:
            Default transition duration in milliseconds
        """
        return self.default_duration_ms
    
    def set_transition_duration_ms(self, duration_ms: float) -> None:
        """
        Set the default transition duration.
        
        Args:
            duration_ms: New default duration (clamped to valid range)
        """
        self.default_duration_ms = max(self.min_duration_ms,
                                       min(self.max_duration_ms, duration_ms))
    
    def is_transition_needed(self, from_segment: Dict, to_segment: Dict) -> bool:
        """
        Determine if a transition is needed between two segments.
        
        Transitions are needed between:
        - consonant and vowel
        - vowel and consonant
        
        Args:
            from_segment: Source segment info
            to_segment: Target segment info
            
        Returns:
            True if a transition should be inserted
        """
        from_type = from_segment.get('type', 'silence')
        to_type = to_segment.get('type', 'silence')
        
        # Transition needed for consonant-vowel or vowel-consonant
        if from_type == 'consonant' and to_type == 'vowel':
            return True
        if from_type == 'vowel' and to_type == 'consonant':
            return True
        
        return False


def generate_transition(from_segment: Dict, to_segment: Dict,
                       fs: int = 16000, duration_ms: float = 30.0) -> np.ndarray:
    """
    Convenience function to generate a transition between segments.
    
    Args:
        from_segment: Source segment info
        to_segment: Target segment info
        fs: Target sampling frequency
        duration_ms: Transition duration in milliseconds
        
    Returns:
        Audio waveform as numpy array
    """
    generator = TransitionGenerator(fs=fs)
    return generator.generate(from_segment, to_segment, duration_ms)


def is_transition_needed(from_segment: Dict, to_segment: Dict) -> bool:
    """
    Convenience function to check if transition is needed.
    
    Args:
        from_segment: Source segment info
        to_segment: Target segment info
        
    Returns:
        True if a transition should be inserted
    """
    generator = TransitionGenerator()
    return generator.is_transition_needed(from_segment, to_segment)
