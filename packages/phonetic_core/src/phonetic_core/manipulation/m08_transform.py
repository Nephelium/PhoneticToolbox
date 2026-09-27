"""M08 direct V2 migration; SRC-PRAAT. Numerical steps retained. No file IO."""
from parselmouth.praat import call

def transform(snd, speed=1.0, pitch_ratio=1.0, pitch_hz=0.0, *, hz_unit='Hz'):
    if abs(speed - 1.0) > 0.01:
        factor = 1.0 / speed
        snd = call(snd, 'Lengthen (overlap-add)', 75.0, 600.0, factor)
    if abs(pitch_ratio - 1.0) > 0.01 or abs(pitch_hz) > 0.01:
        manipulation = call(snd, 'To Manipulation', 0.01, 75, 600)
        pitch_tier = call(manipulation, 'Extract pitch tier')
        if abs(pitch_ratio - 1.0) > 0.01:
            call(pitch_tier, 'Multiply frequencies', snd.xmin, snd.xmax, pitch_ratio)
        if abs(pitch_hz) > 0.01:
            call(pitch_tier, 'Shift frequencies', snd.xmin, snd.xmax, pitch_hz, hz_unit)
        call([pitch_tier, manipulation], 'Replace pitch tier')
        snd = call(manipulation, 'Get resynthesis (overlap-add)')
    return snd
