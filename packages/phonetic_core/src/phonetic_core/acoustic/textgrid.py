"""M01-C source_id SRC-PRAAT: bounded, strict IntervalTier text parser.

File decoding belongs to the adapter. Praat quoted text uses doubled quotes.
Point tiers are explicitly unsupported by this interval analysis contract.
"""
import math
import re
from ..models.associations import Interval, Tier


def parse_textgrid(text, *, max_chars=2_000_000, max_items=100_000, max_tiers=64):
    if len(text)>max_chars: raise ValueError('TextGrid text limit')
    # Strip labelled syntax outside strings, retaining values. Bound tokens while
    # lexing; never use a regex that recursively backtracks through quoted text.
    tokens=[];i=0
    numeric=re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
    def append(value):
        if len(tokens)>=max_items*5+max_tiers*8+8: raise ValueError('TextGrid token limit')
        tokens.append(value)
    while i<len(text):
        c=text[i]
        if c=='"':
            i+=1;parts=[]
            while i<len(text):
                if text[i]=='"':
                    if i+1<len(text) and text[i+1]=='"': parts.append('"');i+=2;continue
                    i+=1;break
                parts.append(text[i]);i+=1
            else: raise ValueError('Unterminated TextGrid string')
            append(('s',''.join(parts)))
        elif text.startswith('<exists>',i): append(('e','exists'));i+=8
        elif c in '+-.0123456789':
            match=numeric.match(text,i)
            if match:
                value=match.group();end=i+len(value)
                # [1] indexes are labels, never data values.
                if i==0 or text[i-1]!='[': append(('n',value))
                i=end
            else: raise ValueError('Invalid TextGrid number')
        else:
            # Labelled lines have their data after '='; item [N]: is syntax.
            if c.isalpha() or c=='_':
                end=i+1
                while end<len(text) and (text[end].isalnum() or text[end] in '_?'): end+=1
                if text[i:end] in ('nan','inf','NaN','Infinity'): raise ValueError('Nonfinite TextGrid number')
                i=end
            else:i+=1
    index=0
    def get(kind):
        nonlocal index
        if index>=len(tokens) or tokens[index][0]!=kind: raise ValueError('Malformed TextGrid structure')
        value=tokens[index][1];index+=1;return value
    def number():
        value=float(get('n'))
        if not math.isfinite(value): raise ValueError('Nonfinite TextGrid boundary')
        return value
    def count(limit):
        value=number()
        if value!=int(value) or not 0<=value<=limit: raise ValueError('TextGrid count limit')
        return int(value)
    def span(parent=None):
        a,b=number(),number()
        if a>b or (parent and not parent[0]<=a<=b<=parent[1]): raise ValueError('Invalid TextGrid range')
        return a,b
    if get('s') not in ('ooTextFile','ooTextFile short') or get('s')!='TextGrid': raise ValueError('Invalid TextGrid header')
    domain=span();get('e');n=count(max_tiers);tiers=[];total=0
    for _ in range(n):
        if get('s')!='IntervalTier': raise ValueError('Only IntervalTier is supported')
        name=get('s');bounds=span(domain);m=count(max_items-total);total+=m;intervals=[];last=bounds[0]
        for _ in range(m):
            a,b=span(bounds);label=get('s')
            if a<last: raise ValueError('Overlapping TextGrid intervals')
            intervals.append(Interval(a,b,label));last=b
        tiers.append(Tier(name,tuple(intervals)))
    if index!=len(tokens): raise ValueError('Trailing TextGrid data')
    return tuple(tiers)
