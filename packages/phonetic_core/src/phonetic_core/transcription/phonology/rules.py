# Direct V2 migration; source_id=PENDING-PHONOLOGY. See NOTICE.md and M14-source-map.md.
from __future__ import annotations
import re
from collections import defaultdict
from .parser import PhonologyInductionParser
from .models import PhonologyInputRow, ParsedPhonologyRow, PhonologyAnalysisResult

class PhonologyRules:
    INITIAL_MANNER_ORDER = [
        "鼻音",
        "塞音",
        "塞擦音",
        "擦音",
        "近音",
        "边音",
        "闪音",
        "颤音",
        "其他",
    ]

    INITIAL_PLACE_ORDER = [
        "双唇",
        "唇齿",
        "舌尖前",
        "舌尖中",
        "舌尖后",
        "舌叶",
        "卷舌",
        "龈后",
        "龈腭",
        "硬腭",
        "软腭",
        "小舌",
        "咽",
        "声门",
        "其他",
    ]

    def __init__(self):
        self._parser = PhonologyInductionParser()

    def analyze(
        self,
        rows: list[PhonologyInputRow],
        consonant_only_as_zero_initial: bool = True,
    ) -> PhonologyAnalysisResult:
        parsed_rows: list[ParsedPhonologyRow] = []
        unique_ipa: set[str] = set()
        for row in rows:
            parsed = self._parser.parse(
                row.ipa,
                consonant_only_as_zero_initial=consonant_only_as_zero_initial,
            )
            parsed_rows.append(
                ParsedPhonologyRow(
                    character=row.character,
                    ipa=row.ipa,
                    note=row.note,
                    initial=parsed.initial,
                    final=parsed.final,
                    tone_value=parsed.tone,
                )
            )
            unique_ipa.add(row.ipa)
        unique_initials = self._sort_symbols({row.initial for row in parsed_rows})
        unique_initials = self._sort_initials(set(unique_initials))
        unique_finals = self._sort_symbols({row.final for row in parsed_rows})
        unique_tones = self._sort_tones({row.tone_value for row in parsed_rows})
        return PhonologyAnalysisResult(
            rows=parsed_rows,
            unique_initials=unique_initials,
            unique_finals=unique_finals,
            unique_tones=unique_tones,
            unique_ipa=sorted(unique_ipa),
        )

    def find_single_consonant_rows(
        self, rows: list[PhonologyInputRow]
    ) -> list[PhonologyInputRow]:
        return [row for row in rows if self._parser.is_single_consonant_syllable(row.ipa)]

    def apply_symbol_aliases(
        self,
        analysis: PhonologyAnalysisResult,
        initial_merge_map: dict[str, str],
        final_merge_map: dict[str, str],
    ) -> PhonologyAnalysisResult:
        def resolve(value: str, merge_map: dict[str, str]) -> str:
            current = value
            visited: set[str] = set()
            while current in merge_map and current not in visited:
                visited.add(current)
                current = merge_map[current]
            return current

        remapped_rows: list[ParsedPhonologyRow] = []
        for row in analysis.rows:
            remapped_rows.append(
                ParsedPhonologyRow(
                    character=row.character,
                    ipa=row.ipa,
                    note=row.note,
                    initial=resolve(row.initial, initial_merge_map),
                    final=resolve(row.final, final_merge_map),
                    tone_value=row.tone_value,
                )
            )
        unique_initials = self._sort_initials({row.initial for row in remapped_rows})
        unique_finals = self._sort_symbols({row.final for row in remapped_rows})
        unique_tones = self._sort_tones({row.tone_value for row in remapped_rows})
        unique_ipa = sorted({row.ipa for row in remapped_rows})
        return PhonologyAnalysisResult(
            rows=remapped_rows,
            unique_initials=unique_initials,
            unique_finals=unique_finals,
            unique_tones=unique_tones,
            unique_ipa=unique_ipa,
        )

    def _parse_columns(self, cols: list[str]) -> PhonologyInputRow | None:
        if not cols:
            return None
        raw_char_col = cols[0].strip() if len(cols) >= 1 else ""
        ipa = cols[1].strip() if len(cols) >= 2 else ""
        note_col = cols[2].strip() if len(cols) >= 3 else ""
        if not raw_char_col and not ipa:
            return None
        if self._is_header_row(raw_char_col, ipa):
            return None
        char, note_from_char = self._extract_character_and_note(raw_char_col)
        note = self._merge_notes(note_from_char, note_col)
        if not char or not ipa:
            return None
        return PhonologyInputRow(character=char, ipa=ipa, note=note)

    def _extract_character_and_note(self, value: str) -> tuple[str, str]:
        raw = value.strip()
        if not raw:
            return "", ""
        chinese_chars = re.findall(r"[\u3400-\u9FFF\U00020000-\U0002A6DF]", raw)
        if not chinese_chars:
            return raw[:1], raw[1:].strip()
        char = chinese_chars[0]
        bracket_matches = re.findall(r"[（(]\s*([^）)]+?)\s*[）)]", raw)
        if bracket_matches:
            return char, "，".join(item.strip() for item in bracket_matches if item.strip())
        if len(chinese_chars) > 1:
            return char, "".join(chinese_chars)
        return char, ""

    def _merge_notes(self, note_a: str, note_b: str) -> str:
        values = [v.strip() for v in [note_a, note_b] if v and v.strip()]
        if not values:
            return ""
        seen: set[str] = set()
        merged: list[str] = []
        for item in values:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
        return "，".join(merged)

    def _is_header_row(self, char_col: str, ipa_col: str) -> bool:
        c = char_col.replace(" ", "")
        p = ipa_col.replace(" ", "")
        header_chars = {"汉字", "字头", "字", "字符"}
        header_ipa = {"音标", "ipa", "拼音"}
        return c.lower() in {h.lower() for h in header_chars} and p.lower() in {
            h.lower() for h in header_ipa
        }

    def _sort_symbols(self, values: set[str]) -> list[str]:
        symbol_list = list(values)
        return sorted(symbol_list, key=lambda x: (x != "Ø", x == "", len(x), x))

    def _sort_initials(self, values: set[str]) -> list[str]:
        symbol_list = list(values)
        return sorted(symbol_list, key=self._initial_sort_key)

    def _sort_tones(self, tones: set[str]) -> list[str]:
        def key(value: str) -> tuple[int, int, str]:
            if value.isdigit():
                return (0, int(value), value)
            return (1, 0, value)

        return sorted(tones, key=key)

    def _resolve_order(self, current: list[str], preferred: list[str] | None) -> list[str]:
        if not preferred:
            return list(current)
        preferred_unique = []
        seen: set[str] = set()
        for item in preferred:
            if item in current and item not in seen:
                preferred_unique.append(item)
                seen.add(item)
        for item in current:
            if item not in seen:
                preferred_unique.append(item)
        return preferred_unique

    def _group_rows(
        self,
        analysis: PhonologyAnalysisResult,
        tone_class_map: dict[str, str],
        outer_key: str,
        inner_key: str,
    ) -> dict[str, dict[str, dict[str, list[ParsedPhonologyRow]]]]:
        grouped: dict[str, dict[str, dict[str, list[ParsedPhonologyRow]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for row in analysis.rows:
            outer = row.final if outer_key == "final" else row.initial
            inner = row.initial if inner_key == "initial" else row.final
            tone_label = tone_class_map.get(row.tone_value, row.tone_value or "0")
            grouped[outer][inner][tone_label].append(row)
        return grouped

    def _initial_sort_key(self, symbol: str):
        if symbol == "Ø":
            return (-1, -1, 0, symbol)
        if symbol == "":
            return (99, 99, 99, symbol)
        manner, place = self._classify_initial(symbol)
        manner_idx = self.INITIAL_MANNER_ORDER.index(manner) if manner in self.INITIAL_MANNER_ORDER else 99
        place_idx = self.INITIAL_PLACE_ORDER.index(place) if place in self.INITIAL_PLACE_ORDER else 99
        return (place_idx, manner_idx, len(symbol), symbol)

    def _classify_initial(self, symbol: str) -> tuple[str, str]:
        s = symbol
        if any(x in s for x in ["m", "n", "ŋ", "ɲ", "ɳ", "ɴ", "ȵ"]):
            manner = "鼻音"
        elif any(x in s for x in ["ts", "tɕ", "dʑ", "tʂ", "ɖʐ", "tʃ", "dʒ"]):
            manner = "塞擦音"
        elif any(x in s for x in ["p", "b", "t", "d", "k", "ɡ", "q", "ɢ", "ʔ", "ȶ", "ȡ"]):
            manner = "塞音"
        elif any(x in s for x in ["s", "z", "ʃ", "ʒ", "ʂ", "ʐ", "ɕ", "ʑ", "f", "v", "x", "ɣ", "h", "ɦ", "χ", "ʁ"]):
            manner = "擦音"
        elif any(x in s for x in ["l", "ɭ", "ʎ", "ʟ"]):
            manner = "边音"
        elif any(x in s for x in ["ɾ", "ɽ", "ɺ"]):
            manner = "闪音"
        elif any(x in s for x in ["r", "ʀ", "ʙ"]):
            manner = "颤音"
        elif any(x in s for x in ["ɹ", "ɻ", "j", "w", "ɰ", "ʋ"]):
            manner = "近音"
        else:
            manner = "其他"

        if any(x in s for x in ["p", "b", "m", "ʘ"]):
            place = "双唇"
        elif any(x in s for x in ["f", "v", "ɱ"]):
            place = "唇齿"
        elif any(x in s for x in ["θ", "ð"]):
            place = "舌尖前"
        elif any(x in s for x in ["t", "d", "n", "s", "z", "l", "ɾ", "r"]):
            place = "舌尖中"
        elif any(x in s for x in ["ʈ", "ɖ", "ɳ", "ʂ", "ʐ", "ɻ"]):
            place = "卷舌"
        elif any(x in s for x in ["ʃ", "ʒ"]):
            place = "龈后"
        elif any(x in s for x in ["ɕ", "ʑ", "ȶ", "ȡ", "ȵ", "tɕ", "dʑ"]):
            place = "龈腭"
        elif any(x in s for x in ["c", "ɟ", "ɲ", "j"]):
            place = "硬腭"
        elif any(x in s for x in ["k", "ɡ", "x", "ɣ", "ŋ", "w", "ɰ"]):
            place = "软腭"
        elif any(x in s for x in ["q", "ɢ", "χ", "ʁ", "ɴ"]):
            place = "小舌"
        elif any(x in s for x in ["ħ", "ʕ", "ʜ", "ʢ"]):
            place = "咽"
        elif any(x in s for x in ["h", "ɦ", "ʔ"]):
            place = "声门"
        else:
            place = "其他"
        return manner, place
