"""
consonant_data.py

Provides consonant acoustic parameters for the Klatt synthesizer.
Contains data classes and dictionaries defining acoustic properties
for all IPA consonants organized by manner of articulation.

Classes:
    ConsonantParams: Data class containing acoustic parameters for a consonant

Constants:
    NASALS: Set of nasal consonant symbols
    PLOSIVES: Set of plosive consonant symbols
    SIBILANTS: Set of sibilant fricative consonant symbols
    FRICATIVES: Set of non-sibilant fricative consonant symbols
    APPROXIMANTS: Set of approximant consonant symbols
    TAPS: Set of tap/flap consonant symbols
    TRILLS: Set of trill consonant symbols
    LATERAL_FRICATIVES: Set of lateral fricative consonant symbols
    LATERAL_APPROXIMANTS: Set of lateral approximant consonant symbols
    LATERAL_FLAPS: Set of lateral flap consonant symbols
    CONSONANT_DATA: Dictionary mapping IPA symbols to ConsonantParams
"""

from dataclasses import dataclass
from typing import Dict, Set


@dataclass
class ConsonantParams:
    """
    Data class containing acoustic parameters for a consonant.
    
    Attributes:
        symbol: IPA symbol for the consonant
        manner: Manner of articulation (nasal, plosive, fricative, etc.)
        place: Place of articulation (bilabial, alveolar, velar, etc.)
        voiced: Whether the consonant is voiced
        f1: First formant frequency in Hz (for resonant consonants)
        f2: Second formant frequency in Hz (for resonant consonants)
        f3: Third formant frequency in Hz (for resonant consonants)
        fnp: Nasal pole frequency in Hz (for nasals only)
        fnz: Nasal zero (anti-formant) frequency in Hz (for nasals only)
        noise_freq: Center frequency of friction noise in Hz (for fricatives)
        noise_bw: Bandwidth of friction noise in Hz (for fricatives)
        default_duration: Default duration in milliseconds
        duration_adjustable: Whether duration can be modified by user
    """
    symbol: str
    manner: str
    place: str
    voiced: bool
    f1: float = 0.0
    f2: float = 0.0
    f3: float = 0.0
    fnp: float = 0.0
    fnz: float = 0.0
    noise_freq: float = 0.0
    noise_bw: float = 0.0
    default_duration: float = 80.0
    duration_adjustable: bool = True


# ============================================================================
# Consonant Category Constants
# ============================================================================

# Nasal consonants
NASALS: Set[str] = {'m', 'ɱ', 'n', 'ɳ', 'ɲ', 'ŋ', 'ɴ'}

# Plosive (stop) consonants
PLOSIVES: Set[str] = {'p', 'b', 't', 'd', 'ʈ', 'ɖ', 'c', 'ɟ', 'k', 'ɡ', 'q', 'ɢ', 'ʡ', 'ʔ'}

# Sibilant fricatives (high-frequency friction noise)
SIBILANTS: Set[str] = {'s', 'z', 'ʃ', 'ʒ', 'ʂ', 'ʐ', 'ɕ', 'ʑ'}

# Non-sibilant fricatives
FRICATIVES: Set[str] = {'ɸ', 'β', 'f', 'v', 'θ', 'ð', 'x', 'ɣ', 'χ', 'ʁ', 'ħ', 'ʕ', 'ʜ', 'ʢ', 'h', 'ɦ'}

# Approximants
APPROXIMANTS: Set[str] = {'ʋ', 'ɹ', 'ɻ', 'j', 'ɰ', 'w'}

# Taps/Flaps
TAPS: Set[str] = {'ⱱ', 'ɾ', 'ɽ'}

# Trills
TRILLS: Set[str] = {'ʙ', 'r', 'ʀ'}

# Lateral fricatives
LATERAL_FRICATIVES: Set[str] = {'ɬ', 'ɮ'}

# Lateral approximants
LATERAL_APPROXIMANTS: Set[str] = {'l', 'ɭ', 'ʎ', 'ʟ'}

# Lateral flaps
LATERAL_FLAPS: Set[str] = {'ɺ'}

# All consonants combined
ALL_CONSONANTS: Set[str] = (
    NASALS | PLOSIVES | SIBILANTS | FRICATIVES | 
    APPROXIMANTS | TAPS | TRILLS | 
    LATERAL_FRICATIVES | LATERAL_APPROXIMANTS | LATERAL_FLAPS
)

# Consonants with adjustable duration
DURATION_ADJUSTABLE: Set[str] = (
    NASALS | SIBILANTS | FRICATIVES | 
    APPROXIMANTS | LATERAL_FRICATIVES | LATERAL_APPROXIMANTS
)

# Consonants with fixed duration (plosives, taps, trills)
DURATION_FIXED: Set[str] = PLOSIVES | TAPS | TRILLS | LATERAL_FLAPS


# ============================================================================
# Consonant Acoustic Parameters Data
# ============================================================================

CONSONANT_DATA: Dict[str, ConsonantParams] = {
    # ========================================================================
    # NASALS - voiced, with nasal pole (FNP) and nasal zero (FNZ)
    # ========================================================================
    'm': ConsonantParams(
        symbol='m', manner='nasal', place='bilabial', voiced=True,
        f1=250, f2=1000, f3=2200, fnp=250, fnz=1000,
        default_duration=80, duration_adjustable=True
    ),
    'ɱ': ConsonantParams(
        symbol='ɱ', manner='nasal', place='labiodental', voiced=True,
        f1=250, f2=1100, f3=2200, fnp=250, fnz=1100,
        default_duration=80, duration_adjustable=True
    ),
    'n': ConsonantParams(
        symbol='n', manner='nasal', place='alveolar', voiced=True,
        f1=250, f2=1400, f3=2200, fnp=250, fnz=1500,
        default_duration=80, duration_adjustable=True
    ),
    'ɳ': ConsonantParams(
        symbol='ɳ', manner='nasal', place='retroflex', voiced=True,
        f1=250, f2=1300, f3=2100, fnp=250, fnz=1400,
        default_duration=80, duration_adjustable=True
    ),
    'ɲ': ConsonantParams(
        symbol='ɲ', manner='nasal', place='palatal', voiced=True,
        f1=250, f2=1800, f3=2500, fnp=250, fnz=1800,
        default_duration=80, duration_adjustable=True
    ),
    'ŋ': ConsonantParams(
        symbol='ŋ', manner='nasal', place='velar', voiced=True,
        f1=250, f2=1800, f3=2200, fnp=250, fnz=2000,
        default_duration=80, duration_adjustable=True
    ),
    'ɴ': ConsonantParams(
        symbol='ɴ', manner='nasal', place='uvular', voiced=True,
        f1=250, f2=1600, f3=2100, fnp=250, fnz=1800,
        default_duration=80, duration_adjustable=True
    ),

    # ========================================================================
    # PLOSIVES - closure + release burst, fixed duration
    # ========================================================================
    'p': ConsonantParams(
        symbol='p', manner='plosive', place='bilabial', voiced=False,
        noise_freq=500, noise_bw=400,
        default_duration=80, duration_adjustable=False
    ),
    'b': ConsonantParams(
        symbol='b', manner='plosive', place='bilabial', voiced=True,
        noise_freq=500, noise_bw=400,
        default_duration=80, duration_adjustable=False
    ),
    't': ConsonantParams(
        symbol='t', manner='plosive', place='alveolar', voiced=False,
        noise_freq=4000, noise_bw=1000,
        default_duration=80, duration_adjustable=False
    ),
    'd': ConsonantParams(
        symbol='d', manner='plosive', place='alveolar', voiced=True,
        noise_freq=4000, noise_bw=1000,
        default_duration=80, duration_adjustable=False
    ),
    'ʈ': ConsonantParams(
        symbol='ʈ', manner='plosive', place='retroflex', voiced=False,
        noise_freq=3500, noise_bw=1000,
        default_duration=80, duration_adjustable=False
    ),
    'ɖ': ConsonantParams(
        symbol='ɖ', manner='plosive', place='retroflex', voiced=True,
        noise_freq=3500, noise_bw=1000,
        default_duration=80, duration_adjustable=False
    ),
    'c': ConsonantParams(
        symbol='c', manner='plosive', place='palatal', voiced=False,
        noise_freq=3000, noise_bw=800,
        default_duration=80, duration_adjustable=False
    ),
    'ɟ': ConsonantParams(
        symbol='ɟ', manner='plosive', place='palatal', voiced=True,
        noise_freq=3000, noise_bw=800,
        default_duration=80, duration_adjustable=False
    ),
    'k': ConsonantParams(
        symbol='k', manner='plosive', place='velar', voiced=False,
        noise_freq=2000, noise_bw=600,
        default_duration=80, duration_adjustable=False
    ),
    'ɡ': ConsonantParams(
        symbol='ɡ', manner='plosive', place='velar', voiced=True,
        noise_freq=2000, noise_bw=600,
        default_duration=80, duration_adjustable=False
    ),
    'q': ConsonantParams(
        symbol='q', manner='plosive', place='uvular', voiced=False,
        noise_freq=1500, noise_bw=500,
        default_duration=80, duration_adjustable=False
    ),
    'ɢ': ConsonantParams(
        symbol='ɢ', manner='plosive', place='uvular', voiced=True,
        noise_freq=1500, noise_bw=500,
        default_duration=80, duration_adjustable=False
    ),
    'ʡ': ConsonantParams(
        symbol='ʡ', manner='plosive', place='epiglottal', voiced=True,
        noise_freq=800, noise_bw=400,
        default_duration=80, duration_adjustable=False
    ),
    'ʔ': ConsonantParams(
        symbol='ʔ', manner='plosive', place='glottal', voiced=False,
        noise_freq=500, noise_bw=300,
        default_duration=80, duration_adjustable=False
    ),

    # ========================================================================
    # SIBILANT FRICATIVES - high-frequency friction noise
    # ========================================================================
    's': ConsonantParams(
        symbol='s', manner='sibilant', place='alveolar', voiced=False,
        noise_freq=6000, noise_bw=2000,
        default_duration=100, duration_adjustable=True
    ),
    'z': ConsonantParams(
        symbol='z', manner='sibilant', place='alveolar', voiced=True,
        noise_freq=6000, noise_bw=2000,
        default_duration=100, duration_adjustable=True
    ),
    'ʃ': ConsonantParams(
        symbol='ʃ', manner='sibilant', place='postalveolar', voiced=False,
        noise_freq=4500, noise_bw=1500,
        default_duration=100, duration_adjustable=True
    ),
    'ʒ': ConsonantParams(
        symbol='ʒ', manner='sibilant', place='postalveolar', voiced=True,
        noise_freq=4500, noise_bw=1500,
        default_duration=100, duration_adjustable=True
    ),
    'ʂ': ConsonantParams(
        symbol='ʂ', manner='sibilant', place='retroflex', voiced=False,
        noise_freq=4000, noise_bw=1500,
        default_duration=100, duration_adjustable=True
    ),
    'ʐ': ConsonantParams(
        symbol='ʐ', manner='sibilant', place='retroflex', voiced=True,
        noise_freq=4000, noise_bw=1500,
        default_duration=100, duration_adjustable=True
    ),
    'ɕ': ConsonantParams(
        symbol='ɕ', manner='sibilant', place='alveolopalatal', voiced=False,
        noise_freq=5500, noise_bw=1800,
        default_duration=100, duration_adjustable=True
    ),
    'ʑ': ConsonantParams(
        symbol='ʑ', manner='sibilant', place='alveolopalatal', voiced=True,
        noise_freq=5500, noise_bw=1800,
        default_duration=100, duration_adjustable=True
    ),

    # ========================================================================
    # NON-SIBILANT FRICATIVES
    # ========================================================================
    'ɸ': ConsonantParams(
        symbol='ɸ', manner='fricative', place='bilabial', voiced=False,
        noise_freq=2000, noise_bw=1000,
        default_duration=100, duration_adjustable=True
    ),
    'β': ConsonantParams(
        symbol='β', manner='fricative', place='bilabial', voiced=True,
        noise_freq=2000, noise_bw=1000,
        default_duration=100, duration_adjustable=True
    ),
    'f': ConsonantParams(
        symbol='f', manner='fricative', place='labiodental', voiced=False,
        noise_freq=3000, noise_bw=1200,
        default_duration=100, duration_adjustable=True
    ),
    'v': ConsonantParams(
        symbol='v', manner='fricative', place='labiodental', voiced=True,
        noise_freq=3000, noise_bw=1200,
        default_duration=100, duration_adjustable=True
    ),
    'θ': ConsonantParams(
        symbol='θ', manner='fricative', place='dental', voiced=False,
        noise_freq=5000, noise_bw=1500,
        default_duration=100, duration_adjustable=True
    ),
    'ð': ConsonantParams(
        symbol='ð', manner='fricative', place='dental', voiced=True,
        noise_freq=5000, noise_bw=1500,
        default_duration=100, duration_adjustable=True
    ),
    'x': ConsonantParams(
        symbol='x', manner='fricative', place='velar', voiced=False,
        noise_freq=2500, noise_bw=1000,
        default_duration=100, duration_adjustable=True
    ),
    'ɣ': ConsonantParams(
        symbol='ɣ', manner='fricative', place='velar', voiced=True,
        noise_freq=2500, noise_bw=1000,
        default_duration=100, duration_adjustable=True
    ),
    'χ': ConsonantParams(
        symbol='χ', manner='fricative', place='uvular', voiced=False,
        noise_freq=2000, noise_bw=800,
        default_duration=100, duration_adjustable=True
    ),
    'ʁ': ConsonantParams(
        symbol='ʁ', manner='fricative', place='uvular', voiced=True,
        noise_freq=2000, noise_bw=800,
        default_duration=100, duration_adjustable=True
    ),
    'ħ': ConsonantParams(
        symbol='ħ', manner='fricative', place='pharyngeal', voiced=False,
        noise_freq=1500, noise_bw=600,
        default_duration=100, duration_adjustable=True
    ),
    'ʕ': ConsonantParams(
        symbol='ʕ', manner='fricative', place='pharyngeal', voiced=True,
        noise_freq=1500, noise_bw=600,
        default_duration=100, duration_adjustable=True
    ),
    'ʜ': ConsonantParams(
        symbol='ʜ', manner='fricative', place='epiglottal', voiced=False,
        noise_freq=1200, noise_bw=500,
        default_duration=100, duration_adjustable=True
    ),
    'ʢ': ConsonantParams(
        symbol='ʢ', manner='fricative', place='epiglottal', voiced=True,
        noise_freq=1200, noise_bw=500,
        default_duration=100, duration_adjustable=True
    ),
    'h': ConsonantParams(
        symbol='h', manner='fricative', place='glottal', voiced=False,
        noise_freq=1000, noise_bw=2000,  # Wide bandwidth for whisper-like quality
        default_duration=80, duration_adjustable=True
    ),
    'ɦ': ConsonantParams(
        symbol='ɦ', manner='fricative', place='glottal', voiced=True,
        noise_freq=1000, noise_bw=2000,
        default_duration=80, duration_adjustable=True
    ),

    # ========================================================================
    # APPROXIMANTS - vowel-like formant structure
    # ========================================================================
    'ʋ': ConsonantParams(
        symbol='ʋ', manner='approximant', place='labiodental', voiced=True,
        f1=300, f2=1100, f3=2400,
        default_duration=60, duration_adjustable=True
    ),
    'ɹ': ConsonantParams(
        symbol='ɹ', manner='approximant', place='alveolar', voiced=True,
        f1=350, f2=1300, f3=1700,  # Low F3 characteristic of English /r/
        default_duration=60, duration_adjustable=True
    ),
    'ɻ': ConsonantParams(
        symbol='ɻ', manner='approximant', place='retroflex', voiced=True,
        f1=350, f2=1200, f3=1600,
        default_duration=60, duration_adjustable=True
    ),
    'j': ConsonantParams(
        symbol='j', manner='approximant', place='palatal', voiced=True,
        f1=280, f2=2200, f3=2800,  # High F2 characteristic of palatal
        default_duration=60, duration_adjustable=True
    ),
    'ɰ': ConsonantParams(
        symbol='ɰ', manner='approximant', place='velar', voiced=True,
        f1=300, f2=1800, f3=2400,
        default_duration=60, duration_adjustable=True
    ),
    'w': ConsonantParams(
        symbol='w', manner='approximant', place='labiovelar', voiced=True,
        f1=300, f2=700, f3=2200,  # Low F2 characteristic of labial
        default_duration=60, duration_adjustable=True
    ),

    # ========================================================================
    # TAPS/FLAPS - very short duration, fixed
    # ========================================================================
    'ⱱ': ConsonantParams(
        symbol='ⱱ', manner='tap', place='labiodental', voiced=True,
        f1=350, f2=1100, f3=2400,
        default_duration=30, duration_adjustable=False
    ),
    'ɾ': ConsonantParams(
        symbol='ɾ', manner='tap', place='alveolar', voiced=True,
        f1=350, f2=1400, f3=2500,
        default_duration=30, duration_adjustable=False
    ),
    'ɽ': ConsonantParams(
        symbol='ɽ', manner='tap', place='retroflex', voiced=True,
        f1=350, f2=1300, f3=2400,
        default_duration=30, duration_adjustable=False
    ),

    # ========================================================================
    # TRILLS - periodic amplitude modulation, fixed duration
    # ========================================================================
    'ʙ': ConsonantParams(
        symbol='ʙ', manner='trill', place='bilabial', voiced=True,
        f1=300, f2=900, f3=2200,
        default_duration=100, duration_adjustable=False
    ),
    'r': ConsonantParams(
        symbol='r', manner='trill', place='alveolar', voiced=True,
        f1=350, f2=1400, f3=2500,
        default_duration=100, duration_adjustable=False
    ),
    'ʀ': ConsonantParams(
        symbol='ʀ', manner='trill', place='uvular', voiced=True,
        f1=350, f2=1200, f3=2300,
        default_duration=100, duration_adjustable=False
    ),

    # ========================================================================
    # LATERAL FRICATIVES - lateral airflow with friction
    # ========================================================================
    'ɬ': ConsonantParams(
        symbol='ɬ', manner='lateral_fricative', place='alveolar', voiced=False,
        f1=350, f2=1200, f3=2800,
        noise_freq=4000, noise_bw=1500,
        default_duration=100, duration_adjustable=True
    ),
    'ɮ': ConsonantParams(
        symbol='ɮ', manner='lateral_fricative', place='alveolar', voiced=True,
        f1=350, f2=1200, f3=2800,
        noise_freq=4000, noise_bw=1500,
        default_duration=100, duration_adjustable=True
    ),

    # ========================================================================
    # LATERAL APPROXIMANTS - lateral airflow, vowel-like
    # ========================================================================
    'l': ConsonantParams(
        symbol='l', manner='lateral_approximant', place='alveolar', voiced=True,
        f1=350, f2=1200, f3=2800,
        default_duration=80, duration_adjustable=True
    ),
    'ɭ': ConsonantParams(
        symbol='ɭ', manner='lateral_approximant', place='retroflex', voiced=True,
        f1=350, f2=1100, f3=2600,
        default_duration=80, duration_adjustable=True
    ),
    'ʎ': ConsonantParams(
        symbol='ʎ', manner='lateral_approximant', place='palatal', voiced=True,
        f1=350, f2=1800, f3=2800,
        default_duration=80, duration_adjustable=True
    ),
    'ʟ': ConsonantParams(
        symbol='ʟ', manner='lateral_approximant', place='velar', voiced=True,
        f1=350, f2=1500, f3=2500,
        default_duration=80, duration_adjustable=True
    ),

    # ========================================================================
    # LATERAL FLAPS - very short lateral, fixed duration
    # ========================================================================
    'ɺ': ConsonantParams(
        symbol='ɺ', manner='lateral_flap', place='alveolar', voiced=True,
        f1=350, f2=1200, f3=2800,
        default_duration=30, duration_adjustable=False
    ),
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_consonant_params(symbol: str) -> ConsonantParams:
    """
    Get the acoustic parameters for a consonant symbol.
    
    Args:
        symbol: IPA consonant symbol
        
    Returns:
        ConsonantParams object for the consonant
        
    Raises:
        KeyError: If the symbol is not found in CONSONANT_DATA
    """
    return CONSONANT_DATA[symbol]


def is_consonant(symbol: str) -> bool:
    """Check if a symbol is a known consonant."""
    return symbol in CONSONANT_DATA


def is_voiced(symbol: str) -> bool:
    """Check if a consonant is voiced."""
    if symbol in CONSONANT_DATA:
        return CONSONANT_DATA[symbol].voiced
    return False


def is_duration_adjustable(symbol: str) -> bool:
    """Check if a consonant's duration can be adjusted."""
    if symbol in CONSONANT_DATA:
        return CONSONANT_DATA[symbol].duration_adjustable
    return False


def get_manner(symbol: str) -> str:
    """Get the manner of articulation for a consonant."""
    if symbol in CONSONANT_DATA:
        return CONSONANT_DATA[symbol].manner
    return ""


def get_place(symbol: str) -> str:
    """Get the place of articulation for a consonant."""
    if symbol in CONSONANT_DATA:
        return CONSONANT_DATA[symbol].place
    return ""


def get_consonants_by_manner(manner: str) -> Set[str]:
    """Get all consonants with a specific manner of articulation."""
    return {sym for sym, params in CONSONANT_DATA.items() if params.manner == manner}


def get_consonants_by_place(place: str) -> Set[str]:
    """Get all consonants with a specific place of articulation."""
    return {sym for sym, params in CONSONANT_DATA.items() if params.place == place}
