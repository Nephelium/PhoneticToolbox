import os
import parselmouth
import numpy as np
from parselmouth.praat import call
from phonetic_toolbox.core.manipulation.synthesis import synthesize_from_pitch
from phonetic_toolbox.core.manipulation.batch_utils import generate_batch_linear

class ManipulationService:
    """
    Service for pitch manipulation and synthesis.
    """
    def __init__(self):
        self.snd = None
        self.snd_path = None
        self.pitch = None
        self.f0 = None
        self.times = None
        
    def load_audio(self, path: str):
        """
        Load an audio file and extract pitch.
        
        Args:
            path (str): Path to the audio file.
            
        Returns:
            tuple: (times, f0, xmin, xmax)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        self.snd_path = path
        self.snd = parselmouth.Sound(path)
        
        # Extract Pitch
        self.pitch = self.snd.to_pitch()
        self.f0 = self.pitch.selected_array['frequency']
        self.times = self.pitch.xs()
        
        # Replace 0 with NaN for plotting if needed, but keeping 0 is fine for logic
        # logic uses 0 as unvoiced usually.
        
        return self.times, self.f0, self.snd.xmin, self.snd.xmax
        
    def get_sound_part(self, xmin, xmax):
        """
        Extract a part of the sound.
        """
        if self.snd is None:
            return None
        return self.snd.extract_part(from_time=xmin, to_time=xmax, preserve_times=True)
        
    def synthesize(self, modified_f0, xmin, xmax, speed=1.0):
        """
        Synthesize sound with modified pitch.
        """
        if self.snd is None:
            raise ValueError("No audio loaded")
            
        return synthesize_from_pitch(self.snd, self.times, modified_f0, xmin, xmax, speed)
        
    def batch_generate(self, times, original_f0, xmin, xmax, 
                       t1, t2, f1_list, f2_list, 
                       knot_points, start_mode, end_mode, knot_modes, 
                       offset_mode=False):
        """
        Generate batch files.
        """
        if self.snd is None:
            raise ValueError("No audio loaded")
            
        return generate_batch_linear(self.snd, self.snd_path, times, original_f0, xmin, xmax,
                                     t1, t2, f1_list, f2_list, 
                                     knot_points, start_mode, end_mode, knot_modes, 
                                     offset_mode)
                                     
    def process_single_file(self, fpath, speed, pitch_ratio, pitch_hz, out_folder):
        """
        Process a single file for batch speed/pitch change.
        """
        try:
            fname = os.path.basename(fpath)
            snd = parselmouth.Sound(fpath)
            
            # 1. Speed (Lengthen)
            if abs(speed - 1.0) > 0.01:
                factor = 1.0 / speed
                snd = call(snd, "Lengthen (overlap-add)", 75.0, 600.0, factor)

            # 2. Pitch (Shift)
            if abs(pitch_ratio - 1.0) > 0.01 or abs(pitch_hz) > 0.01:
                manipulation = call(snd, "To Manipulation", 0.01, 75, 600)
                pitch_tier = call(manipulation, "Extract pitch tier")
                
                # Apply ratio shift
                if abs(pitch_ratio - 1.0) > 0.01:
                    call(pitch_tier, "Multiply frequencies", snd.xmin, snd.xmax, pitch_ratio)
                
                # Apply Hz shift
                if abs(pitch_hz) > 0.01:
                    call(pitch_tier, "Shift frequencies", snd.xmin, snd.xmax, pitch_hz, "Hz")
                
                call([pitch_tier, manipulation], "Replace pitch tier")
                snd = call(manipulation, "Get resynthesis (overlap-add)")

            out_path = os.path.join(out_folder, fname)
            snd.save(out_path, "WAV")
            return out_path
            
        except Exception as e:
            raise e
