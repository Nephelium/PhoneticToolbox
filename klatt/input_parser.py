"""
input_parser.py

Input parser for the Klatt synthesizer that supports both vowels and consonants.
Parses IPA symbols with duration modifiers and generates segment lists.

Classes:
    InputParser: Parses input text into segment dictionaries

The parser recognizes:
- Vowel symbols from VOWEL_FORMANTS
- Consonant symbols from CONSONANT_DATA
- Duration modifiers: +, -, *, /
- Space as silence

Duration modifiers:
- '+': Increase duration by 10% (multiply by 1.1)
- '-': Decrease duration by 10% (multiply by 0.9)
- '*': Double duration (multiply by 2.0)
- '/': Halve duration (multiply by 0.5)

For consonants with duration_adjustable=False (plosives, taps, trills),
duration modifiers are ignored and the default duration is used.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass

# Import consonant data
try:
    from .consonant_data import (
        CONSONANT_DATA, is_consonant, is_duration_adjustable,
        get_manner, get_place
    )
except ImportError:
    from consonant_data import (
        CONSONANT_DATA, is_consonant, is_duration_adjustable,
        get_manner, get_place
    )


# Duration modifier characters
DURATION_MODIFIERS: Set[str] = {'+', '-', '*', '/'}


@dataclass
class ParsedSegment:
    """
    Represents a parsed segment from input text.
    
    Attributes:
        type: Segment type ('vowel', 'consonant', 'silence')
        symbol: IPA symbol or space
        duration_modifier: Duration multiplier (1.0 = no modification)
        base_duration_ms: Base duration in milliseconds (for consonants)
        duration_adjustable: Whether duration can be modified
    """
    type: str
    symbol: str
    duration_modifier: float = 1.0
    base_duration_ms: float = 0.0
    duration_adjustable: bool = True


class InputParser:
    """
    Parser for vowel and consonant input strings.
    
    Parses IPA symbols with optional duration modifiers and generates
    a list of segment dictionaries suitable for synthesis.
    """
    
    def __init__(self, vowel_formants: Optional[Dict] = None):
        """
        Initialize the parser.
        
        Args:
            vowel_formants: Dictionary of vowel symbols to formant values.
                           If None, uses a default set.
        """
        # Default vowel formants (can be overridden)
        self.vowel_formants = vowel_formants or self._get_default_vowel_formants()
        self.vowels: Set[str] = set(self.vowel_formants.keys())
        self.consonants: Set[str] = set(CONSONANT_DATA.keys())
    
    def _get_default_vowel_formants(self) -> Dict:
        """Get default vowel formants dictionary."""
        return {
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
            'ɹ': [400, 1600, 2600],
            'ɻ': [400, 1350, 2200],
        }
    
    def _parse_modifiers(self, text: str, start_index: int) -> Tuple[float, int]:
        """
        Parse duration modifiers starting at the given index.
        
        Args:
            text: Input text
            start_index: Index to start parsing modifiers
            
        Returns:
            Tuple of (modifier_value, end_index)
        """
        modifier = 1.0
        i = start_index
        
        while i < len(text) and text[i] in DURATION_MODIFIERS:
            ch = text[i]
            if ch == '+':
                modifier *= 1.1
            elif ch == '-':
                modifier *= 0.9
            elif ch == '*':
                modifier *= 2.0
            elif ch == '/':
                modifier *= 0.5
            i += 1
        
        return (modifier, i)
    
    def parse(self, text: str) -> List[Dict]:
        """
        Parse input text into a list of segment dictionaries.
        
        Args:
            text: Input string containing IPA symbols and modifiers
            
        Returns:
            List of segment dictionaries with keys:
            - type: 'vowel', 'consonant', or 'silence'
            - symbol: IPA symbol or ' '
            - duration_modifier: Duration multiplier
            - base_duration_ms: Base duration for consonants (0 for vowels)
            - duration_adjustable: Whether duration can be modified
        """
        segments: List[Dict] = []
        i = 0
        
        while i < len(text):
            ch = text[i]
            
            # Handle space as silence
            if ch == ' ':
                segments.append({
                    'type': 'silence',
                    'symbol': ' ',
                    'duration_modifier': 1.0,
                    'base_duration_ms': 0.0,
                    'duration_adjustable': True
                })
                i += 1
                continue
            
            # Handle consonants first (they take priority over vowels for ambiguous symbols)
            if ch in self.consonants:
                modifier, end_idx = self._parse_modifiers(text, i + 1)
                params = CONSONANT_DATA[ch]
                
                # For non-adjustable consonants, ignore modifiers
                if params.duration_adjustable:
                    effective_modifier = modifier
                else:
                    effective_modifier = 1.0
                
                segments.append({
                    'type': 'consonant',
                    'symbol': ch,
                    'duration_modifier': effective_modifier,
                    'base_duration_ms': params.default_duration,
                    'duration_adjustable': params.duration_adjustable
                })
                i = end_idx
                continue
            
            # Handle vowels (after consonants to handle ambiguous symbols like ɹ, ɻ)
            ch_lower = ch.lower()
            if ch_lower in self.vowels:
                modifier, end_idx = self._parse_modifiers(text, i + 1)
                segments.append({
                    'type': 'vowel',
                    'symbol': ch_lower,
                    'duration_modifier': modifier,
                    'base_duration_ms': 0.0,
                    'duration_adjustable': True
                })
                i = end_idx
                continue
            
            # Unknown character - skip
            i += 1
        
        return segments
    
    def parse_to_dataclass(self, text: str) -> List[ParsedSegment]:
        """
        Parse input text into a list of ParsedSegment dataclass instances.
        
        Args:
            text: Input string containing IPA symbols and modifiers
            
        Returns:
            List of ParsedSegment instances
        """
        dict_segments = self.parse(text)
        return [
            ParsedSegment(
                type=seg['type'],
                symbol=seg['symbol'],
                duration_modifier=seg['duration_modifier'],
                base_duration_ms=seg['base_duration_ms'],
                duration_adjustable=seg['duration_adjustable']
            )
            for seg in dict_segments
        ]
    
    def get_effective_duration_ms(self, segment: Dict) -> float:
        """
        Calculate the effective duration for a consonant segment.
        
        Args:
            segment: Segment dictionary from parse()
            
        Returns:
            Effective duration in milliseconds
        """
        if segment['type'] != 'consonant':
            return 0.0
        
        base = segment['base_duration_ms']
        modifier = segment['duration_modifier']
        return base * modifier
    
    def is_vowel(self, symbol: str) -> bool:
        """Check if a symbol is a known vowel."""
        return symbol.lower() in self.vowels
    
    def is_consonant_symbol(self, symbol: str) -> bool:
        """Check if a symbol is a known consonant."""
        return symbol in self.consonants
    
    def is_silence(self, symbol: str) -> bool:
        """Check if a symbol represents silence."""
        return symbol == ' '


def calculate_duration_modifier(modifiers: str) -> float:
    """
    Calculate the cumulative duration modifier from a string of modifier characters.
    
    Args:
        modifiers: String of modifier characters (+, -, *, /)
        
    Returns:
        Cumulative modifier value
    """
    result = 1.0
    for ch in modifiers:
        if ch == '+':
            result *= 1.1
        elif ch == '-':
            result *= 0.9
        elif ch == '*':
            result *= 2.0
        elif ch == '/':
            result *= 0.5
    return result


def apply_duration_modifier(base_duration: float, modifier: float, 
                           adjustable: bool = True) -> float:
    """
    Apply a duration modifier to a base duration.
    
    Args:
        base_duration: Base duration in milliseconds
        modifier: Duration modifier value
        adjustable: Whether the duration can be modified
        
    Returns:
        Effective duration in milliseconds
    """
    if not adjustable:
        return base_duration
    return base_duration * modifier
