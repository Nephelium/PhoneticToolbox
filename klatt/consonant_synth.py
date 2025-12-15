"""
consonant_synth.py

Main consonant synthesizer module that integrates all sub-synthesizers
and provides a unified interface for consonant synthesis.

Classes:
    ConsonantSynthesizer: Main class that dispatches to appropriate sub-synthesizers

Requirements:
    - 12.1: Use independent friction noise generator for fricatives (not tdklatt.py)
    - 12.2: Use independent plosive synthesizer for plosives (not tdklatt.py)
    - 12.3: Use tdklatt.py formant synthesis for approximants and laterals
    - 12.4: Extend tdklatt.py to support anti-formants for nasals
    - 12.5: Correctly concatenate consonant audio with vowel audio
"""

import numpy as np
from scipy.signal import resample_poly
from typing import Optional, Dict, List, Tuple, Union
from math import gcd

try:
    from .consonant_data import (
        CONSONANT_DATA, ConsonantParams,
        NASALS, PLOSIVES, SIBILANTS, FRICATIVES,
        APPROXIMANTS, TAPS, TRILLS,
        LATERAL_FRICATIVES, LATERAL_APPROXIMANTS, LATERAL_FLAPS,
        is_consonant, get_manner
    )
    from .nasal_synth import NasalSynthesizer
    from .plosive_synth import PlosiveSynthesizer
    from .fricative_synth import FricativeSynthesizer
    from .approximant_synth import ApproximantSynthesizer
    from .lateral_synth import LateralSynthesizer
    from .tap_trill_synth import TapTrillSynthesizer
    from .transition_gen import TransitionGenerator
except ImportError:
    from consonant_data import (
        CONSONANT_DATA, ConsonantParams,
        NASALS, PLOSIVES, SIBILANTS, FRICATIVES,
        APPROXIMANTS, TAPS, TRILLS,
        LATERAL_FRICATIVES, LATERAL_APPROXIMANTS, LATERAL_FLAPS,
        is_consonant, get_manner
    )
    from nasal_synth import NasalSynthesizer
    from plosive_synth import PlosiveSynthesizer
    from fricative_synth import FricativeSynthesizer
    from approximant_synth import ApproximantSynthesizer
    from lateral_synth import LateralSynthesizer
    from tap_trill_synth import TapTrillSynthesizer
    from transition_gen import TransitionGenerator


class ConsonantSynthesizer:
    """
    Main consonant synthesizer that integrates all sub-synthesizers.
    
    This class provides a unified interface for synthesizing any consonant
    by dispatching to the appropriate sub-synthesizer based on the consonant's
    manner of articulation.
    
    Sub-synthesizers:
    - NasalSynthesizer: For nasal consonants (m, n, ŋ, etc.)
    - PlosiveSynthesizer: For plosive/stop consonants (p, t, k, etc.)
    - FricativeSynthesizer: For fricative consonants (s, f, x, etc.)
    - ApproximantSynthesizer: For approximant consonants (j, w, ɹ, etc.)
    - LateralSynthesizer: For lateral consonants (l, ɬ, ɮ, etc.)
    - TapTrillSynthesizer: For tap and trill consonants (ɾ, r, etc.)
    
    Attributes:
        fs: Target sampling frequency in Hz
        f0: Default fundamental frequency for voicing
        nasal_synth: NasalSynthesizer instance
        plosive_synth: PlosiveSynthesizer instance
        fricative_synth: FricativeSynthesizer instance
        approximant_synth: ApproximantSynthesizer instance
        lateral_synth: LateralSynthesizer instance
        tap_trill_synth: TapTrillSynthesizer instance
        transition_gen: TransitionGenerator instance
    """
    
    def __init__(self, fs: int = 16000, f0: float = 120.0):
        """
        Initialize the consonant synthesizer with all sub-synthesizers.
        
        Args:
            fs: Target sampling frequency in Hz (default 16000)
            f0: Default fundamental frequency for voicing (default 120 Hz)
        """
        self.fs = fs
        self.f0 = f0
        
        # Initialize all sub-synthesizers
        self.nasal_synth = NasalSynthesizer(fs=fs, f0=f0)
        self.plosive_synth = PlosiveSynthesizer(fs=fs, f0=f0)
        self.fricative_synth = FricativeSynthesizer(fs=fs, f0=f0)
        self.approximant_synth = ApproximantSynthesizer(fs=fs, f0=f0)
        self.lateral_synth = LateralSynthesizer(fs=fs, f0=f0)
        self.tap_trill_synth = TapTrillSynthesizer(fs=fs, f0=f0)
        
        # Initialize transition generator
        self.transition_gen = TransitionGenerator(fs=fs, f0=f0)

    def synthesize(self, consonant: str, duration_ms: float,
                   context: Optional[Dict] = None) -> np.ndarray:
        """
        Synthesize a consonant sound.
        
        Dispatches to the appropriate sub-synthesizer based on the
        consonant's manner of articulation.
        
        Args:
            consonant: IPA consonant symbol
            duration_ms: Duration in milliseconds
            context: Optional context information (preceding/following segments)
            
        Returns:
            Audio waveform as numpy array (normalized to [-1, 1])
            
        Raises:
            ValueError: If the consonant symbol is not recognized
        """
        if consonant not in CONSONANT_DATA:
            raise ValueError(f"Unknown consonant: {consonant}")
        
        params = CONSONANT_DATA[consonant]
        manner = params.manner
        
        # Dispatch to appropriate sub-synthesizer based on manner
        if manner == 'nasal':
            # Requirement 12.4: Use extended tdklatt.py for nasals
            return self.nasal_synth.synthesize(params, duration_ms, context)
        
        elif manner == 'plosive':
            # Requirement 12.2: Use independent plosive synthesizer
            return self.plosive_synth.synthesize(params, duration_ms, context)
        
        elif manner in ('sibilant', 'fricative'):
            # Requirement 12.1: Use independent friction noise generator
            return self.fricative_synth.synthesize(params, duration_ms, context)
        
        elif manner == 'approximant':
            # Requirement 12.3: Use tdklatt.py formant synthesis
            return self.approximant_synth.synthesize(params, duration_ms, context)
        
        elif manner in ('lateral_fricative', 'lateral_approximant', 'lateral_flap'):
            # Requirement 12.3: Use tdklatt.py formant synthesis for laterals
            return self.lateral_synth.synthesize(params, duration_ms, context)
        
        elif manner in ('tap', 'trill'):
            return self.tap_trill_synth.synthesize(params, duration_ms, context)
        
        else:
            # Fallback: try approximant synthesis for unknown manner
            return self.approximant_synth.synthesize(params, duration_ms, context)
    
    def synthesize_with_transitions(self, consonant: str, duration_ms: float,
                                     prev_segment: Optional[Dict] = None,
                                     next_segment: Optional[Dict] = None,
                                     transition_duration_ms: float = 30.0) -> np.ndarray:
        """
        Synthesize a consonant with transitions to adjacent segments.
        
        Generates the consonant audio along with any necessary transitions
        to/from adjacent vowels or consonants.
        
        Args:
            consonant: IPA consonant symbol
            duration_ms: Duration of the consonant in milliseconds
            prev_segment: Previous segment info (type, symbol, formants)
            next_segment: Next segment info (type, symbol, formants)
            transition_duration_ms: Duration of transitions in milliseconds
            
        Returns:
            Audio waveform including transitions (normalized to [-1, 1])
        """
        audio_parts = []
        
        # Create consonant segment info
        consonant_seg = {
            'type': 'consonant',
            'symbol': consonant
        }
        
        # Generate transition from previous segment if needed
        if prev_segment and self.transition_gen.is_transition_needed(prev_segment, consonant_seg):
            trans_in = self.transition_gen.generate(
                prev_segment, consonant_seg, transition_duration_ms)
            if len(trans_in) > 0:
                audio_parts.append(trans_in)
        
        # Generate consonant
        consonant_audio = self.synthesize(consonant, duration_ms)
        if len(consonant_audio) > 0:
            audio_parts.append(consonant_audio)
        
        # Generate transition to next segment if needed
        if next_segment and self.transition_gen.is_transition_needed(consonant_seg, next_segment):
            trans_out = self.transition_gen.generate(
                consonant_seg, next_segment, transition_duration_ms)
            if len(trans_out) > 0:
                audio_parts.append(trans_out)
        
        # Concatenate all parts
        if not audio_parts:
            return np.array([])
        
        return self.concatenate_audio(audio_parts)
    
    def get_consonant_params(self, consonant: str) -> Optional[ConsonantParams]:
        """
        Get the acoustic parameters for a consonant.
        
        Args:
            consonant: IPA consonant symbol
            
        Returns:
            ConsonantParams object or None if not found
        """
        return CONSONANT_DATA.get(consonant)
    
    def is_consonant(self, symbol: str) -> bool:
        """
        Check if a symbol is a known consonant.
        
        Args:
            symbol: IPA symbol
            
        Returns:
            True if the symbol is a consonant
        """
        return is_consonant(symbol)
    
    def get_manner(self, consonant: str) -> str:
        """
        Get the manner of articulation for a consonant.
        
        Args:
            consonant: IPA consonant symbol
            
        Returns:
            Manner of articulation string, or empty string if not found
        """
        return get_manner(consonant)
    
    def get_default_duration(self, consonant: str) -> float:
        """
        Get the default duration for a consonant.
        
        Args:
            consonant: IPA consonant symbol
            
        Returns:
            Default duration in milliseconds, or 80.0 if not found
        """
        if consonant in CONSONANT_DATA:
            return CONSONANT_DATA[consonant].default_duration
        return 80.0
    
    def is_duration_adjustable(self, consonant: str) -> bool:
        """
        Check if a consonant's duration can be adjusted.
        
        Args:
            consonant: IPA consonant symbol
            
        Returns:
            True if duration can be adjusted, False otherwise
        """
        if consonant in CONSONANT_DATA:
            return CONSONANT_DATA[consonant].duration_adjustable
        return False

    # ========================================================================
    # Audio Concatenation Methods (Requirement 12.5)
    # ========================================================================
    
    def concatenate_audio(self, audio_segments: List[np.ndarray],
                          crossfade_ms: float = 5.0) -> np.ndarray:
        """
        Concatenate multiple audio segments with optional crossfade.
        
        Handles:
        - Sample rate matching (all segments should be at self.fs)
        - Crossfade between segments for smooth transitions
        - Normalization to prevent clipping
        
        Requirement 12.5: Correctly concatenate consonant, vowel, and
        transition audio segments.
        
        Args:
            audio_segments: List of audio arrays to concatenate
            crossfade_ms: Duration of crossfade between segments in ms
            
        Returns:
            Concatenated audio array (normalized to [-1, 1])
        """
        if not audio_segments:
            return np.array([])
        
        # Filter out empty segments
        segments = [seg for seg in audio_segments if len(seg) > 0]
        
        if not segments:
            return np.array([])
        
        if len(segments) == 1:
            return self._normalize(segments[0])
        
        # Calculate crossfade samples
        crossfade_samples = int(crossfade_ms * self.fs / 1000)
        
        # If crossfade is too small or segments are too short, just concatenate
        if crossfade_samples < 2:
            result = np.concatenate(segments)
            return self._normalize(result)
        
        # Concatenate with crossfade
        result = self._concatenate_with_crossfade(segments, crossfade_samples)
        
        return self._normalize(result)
    
    def _concatenate_with_crossfade(self, segments: List[np.ndarray],
                                     crossfade_samples: int) -> np.ndarray:
        """
        Concatenate segments with crossfade overlap.
        
        Uses linear crossfade to blend the end of one segment with
        the beginning of the next. Uses a simple approach that builds
        the result incrementally.
        
        Args:
            segments: List of audio arrays
            crossfade_samples: Number of samples for crossfade
            
        Returns:
            Concatenated audio array
        """
        if len(segments) == 0:
            return np.array([])
        
        if len(segments) == 1:
            return segments[0].copy()
        
        # Check if crossfade is feasible - need segments longer than 2*crossfade
        # for middle segments to have content
        min_seg_len = min(len(seg) for seg in segments)
        
        # If crossfade is too large, reduce it
        if crossfade_samples * 2 >= min_seg_len:
            crossfade_samples = max(1, min_seg_len // 4)
        
        if crossfade_samples < 2:
            # Just concatenate without crossfade
            return np.concatenate(segments)
        
        # Build result by processing pairs of segments
        result = segments[0].copy()
        
        for i in range(1, len(segments)):
            curr_seg = segments[i]
            
            # Determine actual crossfade for this junction
            actual_xfade = min(crossfade_samples, len(result), len(curr_seg))
            
            if actual_xfade < 2:
                # Just append
                result = np.concatenate([result, curr_seg])
            else:
                # Get overlapping regions
                prev_end = result[-actual_xfade:]
                curr_start = curr_seg[:actual_xfade]
                
                # Create crossfade
                fade_out = np.linspace(1, 0, actual_xfade)
                fade_in = np.linspace(0, 1, actual_xfade)
                crossfaded = prev_end * fade_out + curr_start * fade_in
                
                # Build new result: previous (minus overlap) + crossfade + rest of current
                new_result = np.concatenate([
                    result[:-actual_xfade],
                    crossfaded,
                    curr_seg[actual_xfade:]
                ])
                result = new_result
        
        return result
    
    def concatenate_segments(self, segments: List[Dict]) -> np.ndarray:
        """
        Concatenate a list of segment dictionaries into audio.
        
        Each segment dictionary should contain:
        - type: 'vowel', 'consonant', 'silence', or 'transition'
        - symbol: IPA symbol (for vowel/consonant)
        - duration_ms: Duration in milliseconds
        - audio: Optional pre-synthesized audio array
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            Concatenated audio array (normalized to [-1, 1])
        """
        audio_parts = []
        
        for i, segment in enumerate(segments):
            seg_type = segment.get('type', 'silence')
            duration_ms = segment.get('duration_ms', 80.0)
            
            # Use pre-synthesized audio if available
            if 'audio' in segment and len(segment['audio']) > 0:
                audio_parts.append(segment['audio'])
                continue
            
            # Otherwise synthesize based on type
            if seg_type == 'consonant':
                symbol = segment.get('symbol', '')
                if symbol and symbol in CONSONANT_DATA:
                    audio = self.synthesize(symbol, duration_ms)
                    if len(audio) > 0:
                        audio_parts.append(audio)
            
            elif seg_type == 'silence':
                # Generate silence
                n_samples = int(duration_ms * self.fs / 1000)
                if n_samples > 0:
                    audio_parts.append(np.zeros(n_samples))
            
            elif seg_type == 'transition':
                # Transitions should be pre-generated
                pass
        
        return self.concatenate_audio(audio_parts)
    
    def resample_to_target(self, audio: np.ndarray, 
                           source_fs: int) -> np.ndarray:
        """
        Resample audio from source sample rate to target sample rate.
        
        Handles sample rate matching for audio from different sources.
        
        Args:
            audio: Input audio array
            source_fs: Source sampling frequency in Hz
            
        Returns:
            Resampled audio at self.fs
        """
        if len(audio) == 0:
            return audio
        
        if source_fs == self.fs:
            return audio
        
        # Use rational resampling
        up = self.fs
        down = source_fs
        g = gcd(up, down)
        up //= g
        down //= g
        
        return resample_poly(audio, up, down)
    
    def _normalize(self, audio: np.ndarray, 
                   target_peak: float = 0.95) -> np.ndarray:
        """
        Normalize audio to prevent clipping.
        
        Scales audio so the peak amplitude is at target_peak.
        
        Args:
            audio: Input audio array
            target_peak: Target peak amplitude (default 0.95)
            
        Returns:
            Normalized audio array
        """
        if len(audio) == 0:
            return audio
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio * (target_peak / max_val)
        return audio
    
    def calculate_total_duration(self, segments: List[Dict]) -> float:
        """
        Calculate the total duration of a segment sequence.
        
        Args:
            segments: List of segment dictionaries with duration_ms
            
        Returns:
            Total duration in milliseconds
        """
        total = 0.0
        for segment in segments:
            if 'audio' in segment and len(segment['audio']) > 0:
                # Calculate duration from audio length
                total += len(segment['audio']) / self.fs * 1000
            else:
                total += segment.get('duration_ms', 0.0)
        return total
    
    def get_expected_samples(self, duration_ms: float) -> int:
        """
        Calculate expected number of samples for a duration.
        
        Args:
            duration_ms: Duration in milliseconds
            
        Returns:
            Expected number of samples
        """
        return int(duration_ms * self.fs / 1000)


# ============================================================================
# Convenience Functions
# ============================================================================

def synthesize_consonant(consonant: str, duration_ms: float,
                         fs: int = 16000, f0: float = 120.0) -> np.ndarray:
    """
    Convenience function to synthesize a single consonant.
    
    Args:
        consonant: IPA consonant symbol
        duration_ms: Duration in milliseconds
        fs: Target sampling frequency
        f0: Fundamental frequency for voicing
        
    Returns:
        Audio waveform as numpy array
    """
    synth = ConsonantSynthesizer(fs=fs, f0=f0)
    return synth.synthesize(consonant, duration_ms)


def concatenate_audio_segments(segments: List[np.ndarray],
                               fs: int = 16000,
                               crossfade_ms: float = 5.0) -> np.ndarray:
    """
    Convenience function to concatenate audio segments.
    
    Args:
        segments: List of audio arrays
        fs: Sampling frequency
        crossfade_ms: Crossfade duration in milliseconds
        
    Returns:
        Concatenated audio array
    """
    synth = ConsonantSynthesizer(fs=fs)
    return synth.concatenate_audio(segments, crossfade_ms)
