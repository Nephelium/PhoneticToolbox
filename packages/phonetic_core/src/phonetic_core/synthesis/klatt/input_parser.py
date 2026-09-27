from dataclasses import dataclass


VOWEL_FORMANTS = {
    "i": [270, 2290, 3010],
    "y": [270, 1800, 2200],
    "ɨ": [290, 1400, 2100],
    "ʉ": [290, 1200, 2100],
    "ɯ": [300, 1100, 2200],
    "u": [300, 870, 2240],
    "ɪ": [390, 1990, 2550],
    "ʏ": [390, 1500, 2100],
    "ʊ": [440, 1020, 2240],
    "e": [390, 2030, 2600],
    "ø": [390, 1550, 2200],
    "ɘ": [390, 1300, 2200],
    "ɵ": [390, 1100, 2200],
    "ɤ": [460, 1100, 2300],
    "o": [460, 800, 2250],
    "ə": [500, 1500, 2500],
    "ɛ": [530, 1840, 2480],
    "œ": [530, 1300, 2200],
    "ɜ": [560, 1350, 2200],
    "ɞ": [560, 1100, 2200],
    "ʌ": [640, 1200, 2400],
    "ɔ": [570, 840, 2410],
    "æ": [660, 1720, 2410],
    "ɐ": [660, 1400, 2300],
    "a": [730, 1090, 2440],
    "ɶ": [730, 1000, 2200],
    "ɑ": [730, 1100, 2400],
    "ɒ": [730, 850, 2300],
    "ɹ": [400, 1600, 2600],
    "ɻ": [400, 1350, 2200],
}


@dataclass
class ParsedSegment:
    type: str
    symbol: str
    duration_modifier: float


def _parse_modifier(text: str, start: int) -> tuple[float, int]:
    value = 1.0
    idx = start
    while idx < len(text):
        ch = text[idx]
        if ch == "+":
            value *= 1.1
        elif ch == "-":
            value *= 0.9
        elif ch == "*":
            value *= 2.0
        elif ch == "/":
            value *= 0.5
        else:
            break
        idx += 1
    return value, idx


def parse_vowel_sequence(text: str) -> list[ParsedSegment]:
    segments: list[ParsedSegment] = []
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch == " ":
            modifier, next_idx = _parse_modifier(text, idx + 1)
            segments.append(ParsedSegment(type="silence", symbol=" ", duration_modifier=modifier))
            idx = next_idx
            continue
        symbol = ch.lower()
        if symbol in VOWEL_FORMANTS:
            modifier, next_idx = _parse_modifier(text, idx + 1)
            segments.append(ParsedSegment(type="vowel", symbol=symbol, duration_modifier=modifier))
            idx = next_idx
            continue
        idx += 1
    return segments
