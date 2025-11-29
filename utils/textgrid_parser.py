from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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

def parse_textgrid(path: Path) -> Optional[TextGrid]:
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
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("xmin =") and i < 10: # Global xmin
            try: tg_xmin = float(get_val(line))
            except: pass
        elif line.startswith("xmax =") and i < 10: # Global xmax
            try: tg_xmax = float(get_val(line))
            except: pass
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
        elif line.startswith("name =") and current_tier:
            current_tier.name = get_val(line)
        elif line.startswith("intervals ["):
            # New interval
            # Read next lines for xmin, xmax, text
            int_xmin = 0.0
            int_xmax = 0.0
            int_text = ""
            
            # Look ahead a few lines
            j = 1
            found_params = 0
            while i + j < len(lines) and found_params < 3:
                subline = lines[i+j]
                if subline.startswith("xmin ="):
                    try: int_xmin = float(get_val(subline))
                    except: pass
                    found_params += 1
                elif subline.startswith("xmax ="):
                    try: int_xmax = float(get_val(subline))
                    except: pass
                    found_params += 1
                elif subline.startswith("text ="):
                    int_text = get_val(subline)
                    found_params += 1
                elif subline.startswith("intervals [") or subline.startswith("item ["):
                    break # Safety break
                j += 1
            current_intervals.append(Interval(int_xmin, int_xmax, int_text))
            i += (j - 1) # Skip processed lines
        
        i += 1
        
    if current_tier:
        current_tier.intervals = current_intervals
        tiers.append(current_tier)
        
    return TextGrid(tg_xmin, tg_xmax, tiers)
