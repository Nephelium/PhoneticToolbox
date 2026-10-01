"""Pure M01 interval planning; v2 int(t*fs) and silent-label policy preserved.

Parameter slicing is the explicit M01-MAN01 addition: no re-estimation, interpolation
or normalization. File creation, names, process limits and publication are adapters.
"""
from dataclasses import dataclass
import math
from .models.associations import Tier


@dataclass(frozen=True)
class Segment:
    index: int
    label: str
    start_s: float
    end_s: float
    first: int
    last: int


def plan_segments(tiers,layer,sample_rate,frames,*,max_segments=1000):
    if type(sample_rate)!=int or sample_rate<=0 or type(frames)!=int or frames<=0:
        raise ValueError('Invalid audio dimensions')
    if type(max_segments)!=int or not 1<=max_segments<=1000:raise ValueError('Invalid segment limit')
    matches=[t for t in tiers if isinstance(t,Tier) and t.name==layer]
    if len(matches)!=1:raise ValueError('Missing or ambiguous TextGrid tier')
    segments=[];previous=0.
    for i,interval in enumerate(matches[0].intervals):
        start,end=interval.xmin,interval.xmax
        if not all(type(x) in (int,float) and math.isfinite(x) for x in (start,end)) or not 0<=start<end or start<previous:
            raise ValueError('Invalid or overlapping TextGrid interval')
        previous=end
        label=interval.text.strip()
        if label.lower() in ('','sil','eps','<sil>','<eps>'):continue
        # A TextGrid may retain an empty tail beyond a trimmed WAV. Only an
        # interval that will be exported must lie inside the audio. Never clip
        # labelled intervals or relax the tier's ordering/overlap validation.
        if end>frames/sample_rate:raise ValueError('Labelled interval exceeds audio')
        first,last=int(start*sample_rate),int(end*sample_rate)
        if not 0<=first<last<=frames:raise ValueError('Empty or invalid sample range')
        if len(segments)>=max_segments:raise ValueError('Segment count limit exceeded')
        segments.append(Segment(i,label,float(start),float(end),first,last))
    return tuple(segments)


def slice_parameter_table(table,segment,sample_rate):
    """Slice a validated parent table at sample bounds; preserve absent/Inf values."""
    columns=table['columns'];ti=columns.index('Time_s')
    if 'Source_Time_s' in columns:raise ValueError('Already segmented parameter table')
    start,end=segment.first/sample_rate,segment.last/sample_rate
    indices=[i for i in range(len(columns)) if i!=ti]
    rows=[[row[ti]-start,row[ti],*(row[i] for i in indices)] for row in table['rows'] if start<=row[ti]<end]
    if not rows:return None
    return {'columns':['Time_s','Source_Time_s',*(columns[i] for i in indices)],
            'kinds':['number','number',*(table['kinds'][i] for i in indices)],'rows':rows}
