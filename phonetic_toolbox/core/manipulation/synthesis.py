import parselmouth
from parselmouth.praat import call
import numpy as np

def synthesize_from_pitch(snd, times, modified_f0, xmin, xmax, speed=1.0):
    """
    Synthesize a new sound from modified pitch data within a time range.
    
    Args:
        snd (parselmouth.Sound): The original sound object.
        times (np.array): Time points of the pitch track.
        modified_f0 (np.array): Modified F0 values.
        xmin (float): Start time of the segment to synthesize.
        xmax (float): End time of the segment to synthesize.
        speed (float): Speed factor (1.0 = original speed).
        
    Returns:
        parselmouth.Sound: The synthesized sound.
    """
    part_snd = snd.extract_part(from_time=xmin, to_time=xmax, preserve_times=True)
    new_pitch_tier = call("Create PitchTier", "modified", xmin, xmax)
    
    mask = (times >= xmin) & (times <= xmax)
    part_times = times[mask]
    part_f0 = modified_f0[mask]
    
    for t, f in zip(part_times, part_f0):
        if f > 0:
            call(new_pitch_tier, "Add point", t, f)
    
    manipulation = call(part_snd, "To Manipulation", 0.01, 75, 600)
    call([manipulation, new_pitch_tier], "Replace pitch tier")
    
    # Speed Logic
    if abs(speed - 1.0) > 0.01:
        factor = 1.0 / speed
        duration_tier = call("Create DurationTier", "duration", xmin, xmax)
        call(duration_tier, "Add point", xmin, factor)
        call([manipulation, duration_tier], "Replace duration tier")

    synth_snd = call(manipulation, "Get resynthesis (overlap-add)")
    return synth_snd
