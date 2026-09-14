"""SRC-PRAAT. M12 bounded long/short TextGrid parser, including point tiers.

Lexer follows the existing M01 parser. This separate document contract retains
domain and point tiers, while M01's interval-analysis contract remains unchanged.
"""
import math
import re


def parse_document(text, *, max_chars=2_000_000, max_items=100_000, max_tiers=64):
    if not isinstance(text, str) or len(text) > max_chars:
        raise ValueError('TextGrid text limit')
    tokens = []
    i = 0
    numeric = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
    while i < len(text):
        c = text[i]
        if c == '"':
            i += 1
            parts = []
            while i < len(text):
                if text[i] == '"':
                    if i+1 < len(text) and text[i+1] == '"':
                        parts.append('"'); i += 2; continue
                    i += 1; break
                parts.append(text[i]); i += 1
            else:
                raise ValueError('Unterminated TextGrid string')
            tokens.append(('s', ''.join(parts)))
        elif text.startswith('<exists>', i):
            tokens.append(('e', 'exists')); i += 8
        elif c in '+-.0123456789':
            match = numeric.match(text, i)
            if not match:
                raise ValueError('Invalid TextGrid number')
            if i == 0 or text[i-1] != '[':
                tokens.append(('n', match[0]))
            i = match.end()
        elif c.isalpha() or c == '_':
            end = i+1
            while end < len(text) and (text[end].isalnum() or text[end] in '_?'):
                end += 1
            if text[i:end].lower() in ('nan', 'inf', 'infinity'):
                raise ValueError('Nonfinite TextGrid number')
            i = end
        else:
            i += 1
        if len(tokens) > max_items*5 + max_tiers*8 + 8:
            raise ValueError('TextGrid token limit')
    index = 0

    def get(kind):
        nonlocal index
        if index >= len(tokens) or tokens[index][0] != kind:
            raise ValueError('Malformed TextGrid structure')
        value = tokens[index][1]; index += 1
        return value

    def number():
        value = float(get('n'))
        if not math.isfinite(value):
            raise ValueError('Nonfinite TextGrid boundary')
        return value

    def count(limit):
        value = number()
        if value != int(value) or not 0 <= value <= limit:
            raise ValueError('TextGrid count limit')
        return int(value)

    def span(parent=None):
        a, b = number(), number()
        if a > b or (parent and not parent[0] <= a <= b <= parent[1]):
            raise ValueError('Invalid TextGrid range')
        return a, b

    if get('s') not in ('ooTextFile', 'ooTextFile short') or get('s') != 'TextGrid':
        raise ValueError('Invalid TextGrid header')
    domain = span(); get('e'); n = count(max_tiers)
    if domain[0] < 0 or domain[1] <= domain[0] or n == 0:
        raise ValueError('Invalid TextGrid domain')
    tiers = []; names = set(); total = 0
    for _ in range(n):
        kind, name = get('s'), get('s')
        bounds = span(domain); size = count(max_items-total); total += size
        if name in names:
            raise ValueError('Duplicate TextGrid tier')
        names.add(name)
        tier = dict(name=name, xmin=bounds[0], xmax=bounds[1]); last = bounds[0]
        if kind == 'IntervalTier':
            tier['intervals'] = []
            for _ in range(size):
                a, b = span(bounds); label = get('s')
                if a < last or b <= a:
                    raise ValueError('Overlapping or empty TextGrid intervals')
                tier['intervals'].append(dict(xmin=a, xmax=b, text=label)); last = b
        elif kind == 'TextTier':
            tier['points'] = []
            for _ in range(size):
                value, mark = number(), get('s')
                if not last <= value <= bounds[1]:
                    raise ValueError('Invalid TextGrid point')
                tier['points'].append(dict(number=value, mark=mark)); last = value
        else:
            raise ValueError('Unsupported TextGrid tier')
        tiers.append(tier)
    if index != len(tokens):
        raise ValueError('Trailing TextGrid data')
    return dict(xmin=domain[0], xmax=domain[1], tiers=tiers)
