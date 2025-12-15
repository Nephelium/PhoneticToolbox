from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re

@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str

@dataclass
class Tier:
    name: str
    xmin: float
    xmax: float
    intervals: List[Interval]

@dataclass
class TextGrid:
    xmin: float
    xmax: float
    tiers: List[Tier]


def write_textgrid(tg: TextGrid, path: Path) -> None:
    """
    Write a TextGrid data structure to a file in long format.
    
    Args:
        tg: TextGrid data structure to write
        path: Path to output file
    """
    lines = []
    lines.append('File type = "ooTextFile"')
    lines.append('Object class = "TextGrid"')
    lines.append('')
    lines.append(f'xmin = {tg.xmin}')
    lines.append(f'xmax = {tg.xmax}')
    lines.append('tiers? <exists>')
    lines.append(f'size = {len(tg.tiers)}')
    lines.append('item []:')
    
    for tier_idx, tier in enumerate(tg.tiers, start=1):
        lines.append(f'    item [{tier_idx}]:')
        lines.append('        class = "IntervalTier"')
        lines.append(f'        name = "{tier.name}"')
        lines.append(f'        xmin = {tier.xmin}')
        lines.append(f'        xmax = {tier.xmax}')
        lines.append(f'        intervals: size = {len(tier.intervals)}')
        
        for int_idx, interval in enumerate(tier.intervals, start=1):
            lines.append(f'        intervals [{int_idx}]:')
            lines.append(f'            xmin = {interval.xmin}')
            lines.append(f'            xmax = {interval.xmax}')
            # Escape quotes in text
            escaped_text = interval.text.replace('"', '""')
            lines.append(f'            text = "{escaped_text}"')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _parse_short_format(lines: List[str]) -> Optional[TextGrid]:
    """
    Parse TextGrid in short format.
    
    Short format has values on separate lines without labels.
    """
    try:
        # Skip header lines (File type, Object class, empty line)
        idx = 0
        while idx < len(lines) and not lines[idx].replace('"', '').replace(' ', '').replace('.', '').replace('-', '').lstrip('-').isdigit():
            idx += 1
        
        if idx >= len(lines):
            return None
            
        tg_xmin = float(lines[idx].strip())
        idx += 1
        tg_xmax = float(lines[idx].strip())
        idx += 1
        
        # Skip <exists> line
        if idx < len(lines) and '<exists>' in lines[idx]:
            idx += 1
        
        # Number of tiers
        n_tiers = int(lines[idx].strip())
        idx += 1
        
        tiers = []
        for _ in range(n_tiers):
            # Tier class (IntervalTier or TextTier)
            tier_class = lines[idx].strip().strip('"')
            idx += 1
            
            # Tier name
            tier_name = lines[idx].strip().strip('"')
            idx += 1
            
            # Tier xmin, xmax
            tier_xmin = float(lines[idx].strip())
            idx += 1
            tier_xmax = float(lines[idx].strip())
            idx += 1
            
            # Number of intervals
            n_intervals = int(lines[idx].strip())
            idx += 1
            
            intervals = []
            for _ in range(n_intervals):
                int_xmin = float(lines[idx].strip())
                idx += 1
                int_xmax = float(lines[idx].strip())
                idx += 1
                int_text = lines[idx].strip().strip('"')
                idx += 1
                intervals.append(Interval(int_xmin, int_xmax, int_text))
            
            tiers.append(Tier(tier_name, tier_xmin, tier_xmax, intervals))
        
        return TextGrid(tg_xmin, tg_xmax, tiers)
    except (ValueError, IndexError):
        return None


def _is_short_format(lines: List[str]) -> bool:
    """
    Detect if TextGrid is in short format.
    
    Short format doesn't have "=" signs for assignments.
    """
    # Check first few content lines after header
    for i, line in enumerate(lines[3:10], start=3):
        stripped = line.strip()
        if stripped and '=' not in stripped and stripped not in ['', '<exists>']:
            # If we find a numeric value without "=", it's short format
            try:
                float(stripped)
                return True
            except ValueError:
                pass
    return False


def parse_textgrid(path: Path) -> Optional[TextGrid]:
    """
    Parse a TextGrid file (supports both long and short formats).
    
    Args:
        path: Path to the TextGrid file
        
    Returns:
        TextGrid data structure or None if parsing fails
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()]
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='utf-16') as f:
                lines = [l.strip() for l in f.readlines()]
        except Exception:
            return None
    except Exception:
        return None

    if not lines or lines[0] != 'File type = "ooTextFile"':
        return None

    # Detect and handle short format
    if _is_short_format(lines):
        return _parse_short_format(lines)

    # Parse long format
    return _parse_long_format(lines)


def _parse_long_format(lines: List[str]) -> Optional[TextGrid]:
    """
    Parse TextGrid in long format (with "key = value" syntax).
    """
    # Helper to extract value
    def get_val(line: str) -> str:
        if "=" in line:
            return line.split("=", 1)[1].strip().strip('"')
        return ""

    tg_xmin = 0.0
    tg_xmax = 0.0
    tiers = []
    
    # Simple state machine
    current_tier = None
    current_intervals = []
    in_tier_header = False  # Track if we're reading tier header (before intervals)
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("xmin =") and i < 10:  # Global xmin
            try:
                tg_xmin = float(get_val(line))
            except:
                pass
        elif line.startswith("xmax =") and i < 10:  # Global xmax
            try:
                tg_xmax = float(get_val(line))
            except:
                pass
        elif line.startswith("item ["):
            if line == "item []:":
                pass
            else:
                # New item (tier)
                if current_tier:
                    current_tier.intervals = current_intervals
                    tiers.append(current_tier)
                current_tier = Tier(name="", xmin=0, xmax=0, intervals=[])
                current_intervals = []
                in_tier_header = True
        elif line.startswith("name =") and current_tier:
            current_tier.name = get_val(line)
        elif line.startswith("xmin =") and current_tier and in_tier_header:
            try:
                current_tier.xmin = float(get_val(line))
            except:
                pass
        elif line.startswith("xmax =") and current_tier and in_tier_header:
            try:
                current_tier.xmax = float(get_val(line))
            except:
                pass
        elif line.startswith("intervals:") or line.startswith("intervals ["):
            in_tier_header = False
            if line.startswith("intervals ["):
                # New interval - read next lines for xmin, xmax, text
                int_xmin = 0.0
                int_xmax = 0.0
                int_text = ""
                
                # Look ahead a few lines
                j = 1
                found_params = 0
                while i + j < len(lines) and found_params < 3:
                    subline = lines[i + j]
                    if subline.startswith("xmin ="):
                        try:
                            int_xmin = float(get_val(subline))
                        except:
                            pass
                        found_params += 1
                    elif subline.startswith("xmax ="):
                        try:
                            int_xmax = float(get_val(subline))
                        except:
                            pass
                        found_params += 1
                    elif subline.startswith("text ="):
                        int_text = get_val(subline)
                        found_params += 1
                    elif subline.startswith("intervals [") or subline.startswith("item ["):
                        break  # Safety break
                    j += 1
                current_intervals.append(Interval(int_xmin, int_xmax, int_text))
                i += (j - 1)  # Skip processed lines
        
        i += 1
        
    if current_tier:
        current_tier.intervals = current_intervals
        tiers.append(current_tier)
        
    return TextGrid(tg_xmin, tg_xmax, tiers)
