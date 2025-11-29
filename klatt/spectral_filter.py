import numpy as np
from scipy.signal import stft, istft
from scipy.interpolate import interp1d

from scipy.ndimage import uniform_filter1d

class SpectralFilter:
    def __init__(self, fs):
        self.fs = fs

    def apply_agc(self, audio, target_rms=0.1, window_ms=50):
        """
        Apply Automatic Gain Control to stabilize amplitude.
        """
        if len(audio) == 0: return audio
        
        # Envelope follower (RMS)
        win_len = int(window_ms * self.fs / 1000)
        win_len = max(1, win_len)
        
        s_sq = audio**2
        env = np.sqrt(uniform_filter1d(s_sq, size=win_len) + 1e-10)
        
        # Calculate gain
        # Avoid dividing by zero or amplifying noise too much
        # Threshold for noise floor
        noise_floor = 1e-4
        
        gain = np.zeros_like(env)
        mask = env > noise_floor
        gain[mask] = target_rms / env[mask]
        gain[~mask] = 1.0 # Do nothing for silence
        
        # Smooth gain to avoid artifacts
        # Use a larger window for smoothing gain changes
        smooth_win = int(0.1 * self.fs) # 100ms
        gain_smooth = uniform_filter1d(gain, size=smooth_win)
        
        # Hard limit on gain to prevent explosions
        gain_smooth = np.clip(gain_smooth, 0.0, 100.0)
        
        return audio * gain_smooth

    def apply_shimmer(self, audio, shimmer_arr):
        """
        Apply Shimmer as Amplitude Modulation (Post-processing).
        shimmer_arr: Array of shimmer values in %.
        """
        if len(audio) != len(shimmer_arr):
            # Resize shimmer_arr
            x_old = np.linspace(0, 1, len(shimmer_arr))
            x_new = np.linspace(0, 1, len(audio))
            f = interp1d(x_old, shimmer_arr, kind='linear', fill_value="extrapolate")
            shimmer_arr = f(x_new)
            
        # Noise generator: Uniform [-1, 1]
        noise = np.random.uniform(-1.0, 1.0, size=len(audio))
        
        # Modulation: 1 + (noise * percentage / 100)
        # Shimmer is usually mean abs diff, but for synthesis we use it as modulation depth.
        mod = 1.0 + noise * (shimmer_arr / 100.0)
        
        return audio * mod

    def apply_jitter(self, audio, jitter_arr, f0_arr):
        """
        Apply Jitter as Time Domain Resampling (Post-processing).
        jitter_arr: Array of jitter values in %.
        f0_arr: Array of F0 values (Hz).
        """
        n = len(audio)
        if len(jitter_arr) != n:
            x_old = np.linspace(0, 1, len(jitter_arr))
            x_new = np.linspace(0, 1, n)
            f = interp1d(x_old, jitter_arr, kind='linear', fill_value="extrapolate")
            jitter_arr = f(x_new)
            
        if len(f0_arr) != n:
            x_old = np.linspace(0, 1, len(f0_arr))
            x_new = np.linspace(0, 1, n)
            f = interp1d(x_old, f0_arr, kind='linear', fill_value="extrapolate")
            f0_arr = f(x_new)

        # Time vector
        t = np.arange(n) / self.fs
        
        # Jitter shift calculation
        # Jitter % is relative to period T = 1/F0.
        # Shift amount = random * (Jitter/100) * (1/F0)
        noise = np.random.uniform(-1.0, 1.0, size=n)
        
        # Max shift in seconds
        max_shift = (jitter_arr / 100.0) * (1.0 / (f0_arr + 1e-5))
        
        # Actual shift
        shift = noise * max_shift
        
        # New time points: t' = t + shift
        # Audio_out(t) = Audio_in(t + shift(t))
        t_sample = t + shift
        
        # Clamp to valid range
        t_sample = np.clip(t_sample, 0, (n - 1) / self.fs)
        
        # Interpolate
        x_indices = t_sample * self.fs
        audio_out = np.interp(x_indices, np.arange(n), audio)
        
        return audio_out

    def normalize(self, audio, target_db=-1.0):
        """
        Normalize peak amplitude to target_db (or linear range -1 to 1 if target_db is None).
        User asked for -1~1, so we map max peak to 1.0.
        """
        mx = np.max(np.abs(audio))
        if mx > 1e-8:
            return audio / mx
        return audio

    def process(self, audio, f0_contour, h1h2_contour, slope_contour, hnr_contour):
        """
        Apply frequency domain filtering for H1-H2, Slope, and HNR.
        
        Args:
            audio: 1D numpy array of waveform.
            f0_contour: 1D array of F0 values (time-aligned with STFT frames ideally, or sampled).
            h1h2_contour: 1D array of target H1-H2 values (dB).
            slope_contour: 1D array of target Slope values (dB).
            hnr_contour: 1D array of HNR parameter values (dB).
            
        Returns:
            Filtered audio waveform.
        """
        # STFT parameters
        # 25ms window, 5ms hop (match VoiceSauce analysis usually, or just good default)
        nperseg = int(0.025 * self.fs)
        noverlap = int(0.020 * self.fs) # 5ms hop
        # nfft should be power of 2 for speed, or at least >= nperseg
        nfft = 2**int(np.ceil(np.log2(nperseg)))
        
        f, t, Zxx = stft(audio, fs=self.fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        
        # Zxx is complex: freq bins x time frames
        # Modifying magnitude, keeping phase
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Log magnitude (dB)
        # Avoid log(0)
        log_mag = 20 * np.log10(magnitude + 1e-10)
        
        # Interpolate contours to match STFT time frames
        # Input contours might be at different rate or same rate if generated by parameter array
        # Parameter arrays in GUI are usually generated at self.fs or specific rate?
        # In GUI: arrays[name] = param.get_array(duration, fs) -> sample-wise array!
        # We need to downsample contours to match STFT frames
        num_frames = Zxx.shape[1]
        t_frames = t
        
        # Input contours are likely sample-wise (len = len(audio))
        # We pick values at frame centers
        frame_indices = (t_frames * self.fs).astype(int)
        frame_indices = np.clip(frame_indices, 0, len(f0_contour)-1)
        
        f0_frames = f0_contour[frame_indices]
        h1h2_frames = h1h2_contour[frame_indices]
        slope_frames = slope_contour[frame_indices]
        hnr_frames = hnr_contour[frame_indices]
        
        # Frequency axis
        freqs = f
        
        # Modify frames
        for i in range(num_frames):
            f0 = f0_frames[i]
            if f0 <= 0: continue # Skip unvoiced/silence processing if F0 is invalid
            
            curr_log_mag = log_mag[:, i]
            
            # --- 1. H1-H2 Adjustment ---
            # Anchor: H2
            # Target H1 = H2 + Target_H1H2
            # Gain = Target H1 - Current H1
            
            # Find H1 and H2 indices
            # Search window: +/- 10% or +/- 20Hz?
            idx_h1 = self._find_peak(curr_log_mag, freqs, f0, search_ratio=0.2)
            idx_h2 = self._find_peak(curr_log_mag, freqs, 2*f0, search_ratio=0.2)
            
            h1h2_gain_mask = np.zeros_like(freqs)
            
            if idx_h1 is not None and idx_h2 is not None:
                mag_h1 = curr_log_mag[idx_h1]
                mag_h2 = curr_log_mag[idx_h2]
                
                target_h1h2 = h1h2_frames[i]
                target_mag_h1 = mag_h2 + target_h1h2
                
                required_gain = target_mag_h1 - mag_h1
                
                # Apply to 0 - 1.6 F0
                # 0 - 1.3 F0: Full gain
                # 1.3 - 1.6 F0: Linear decay to 0
                
                mask_full = freqs <= 1.3 * f0
                mask_trans = (freqs > 1.3 * f0) & (freqs <= 1.6 * f0)
                
                h1h2_gain_mask[mask_full] = required_gain
                
                # Transition
                f_trans = freqs[mask_trans]
                if len(f_trans) > 0:
                    # ratio 1.0 at 1.3F0 -> 0.0 at 1.6F0
                    # range width = 0.3 F0
                    # dist from 1.3 F0
                    dist = f_trans - 1.3 * f0
                    ratio = 1.0 - (dist / (0.3 * f0))
                    h1h2_gain_mask[mask_trans] = required_gain * ratio
            
            # --- 2. Spectral Slope Adjustment ---
            # Regression on harmonics -> Current Slope
            # Diff = Target - Current
            # Mask = Diff * freq
            
            # Identify harmonics
            # We can use peaks near n*F0
            harmonics_freqs = []
            harmonics_mags = []
            
            n = 1
            while True:
                target_f = n * f0
                if target_f > freqs[-1] or target_f > 5000: # Limit to 5k for regression? 
                    # User says "adjustment... linear change cutoff at 5000Hz".
                    # Regression should probably use valid harmonics up to some point.
                    # Let's use up to 4000-5000Hz for estimation.
                    break
                
                pidx = self._find_peak(curr_log_mag, freqs, target_f, search_ratio=0.15)
                if pidx is not None:
                    harmonics_freqs.append(freqs[pidx])
                    harmonics_mags.append(curr_log_mag[pidx])
                n += 1
            
            slope_gain_mask = np.zeros_like(freqs)
            
            if len(harmonics_freqs) >= 2:
                # Linear regression: Mag = Slope * Freq + Intercept
                # Freq in Hz? Or kHz? Usually dB/kHz is easier to read.
                # Let's use kHz for numerical stability and common units.
                x = np.array(harmonics_freqs) / 1000.0
                y = np.array(harmonics_mags)
                
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                
                current_slope = m # dB/kHz
                
                # Target Slope from parameter
                # Assuming parameter is also dB/kHz? 
                # If parameter range is -50 to 20, dB/kHz fits well (e.g. -6dB/oct is ~-20dB/dec ~ -6dB/oct).
                # Spectral tilt is often negative.
                target_slope = slope_frames[i] 
                
                slope_diff = target_slope - current_slope # dB/kHz
                
                # Construct mask
                # "adjustment increases linearly with frequency... cutoff at 5000Hz"
                # Updated logic:
                # 0-5000Hz: Adjustment(f) = slope_diff * (f / 1000.0)
                # >5000Hz: Adjustment(f) = Adj(5000) + (slope_diff / 2) * ((f - 5000) / 1000.0)
                
                mask_low = freqs <= 5000
                mask_high = freqs > 5000
                
                # 0-5000Hz
                slope_gain_mask[mask_low] = slope_diff * (freqs[mask_low] / 1000.0)
                
                # >5000Hz
                adj_5k = slope_diff * (5000.0 / 1000.0)
                slope_gain_mask[mask_high] = adj_5k + (slope_diff / 2.0) * ((freqs[mask_high] - 5000.0) / 1000.0)
                
            
            # --- 3. HNR Adjustment ---
            # Parameter range: -70 (clean) to -10 (noisy)
            # We map this to a gain boost for the gaps.
            # -70 dB -> 0 dB Gain
            # -10 dB -> 60 dB Gain (approx)
            
            hnr_val = hnr_frames[i]
            
            # Heuristic mapping: HNR=60 -> Gain=0, HNR=0 -> Gain=Max (e.g. 20dB)
            # Or use HNR value to control visibility of noise?
            # User: "apply positive gain... to reduce HNR".
            # If HNR parameter is high, we don't want to reduce it.
            # If HNR parameter is low (noisy), we want to add noise.
            # Max Gain to apply to noise floor?
            # Let's say max gain is 25dB when HNR=0.
            noise_gain_max = 25.0
            # Clamp HNR to 0-60
            # User requested raw values and "noisy version".
            # If HNR is negative (e.g. -10), gain should be > noise_gain_max.
            # Formula: Gain = Max * (1 - HNR/60).
            # If HNR=60 -> Gain=0.
            # If HNR=0 -> Gain=Max.
            # If HNR=-10 -> Gain=Max * (1 + 10/60) ~ 1.16*Max.
            
            current_noise_gain = noise_gain_max * (1.0 - hnr_val / 60.0)
            # Ensure non-negative gain
            current_noise_gain = max(0.0, current_noise_gain)
            
            hnr_gain_mask = np.zeros_like(freqs)
            
            if current_noise_gain > 0.1:
                # Identify gap regions
                # Centers: 1.5, 2.5, 3.5 ... until Nyquist
                
                # Vectorized mask creation for speed?
                # Frequency grid
                # Gaps are periodic.
                # (freq / f0) % 1.0 approx 0.5
                # gap_bw = 0.2 * f0
                # dist from nearest integer harmonic
                # harmonic_dist = abs(freq - round(freq/f0)*f0)
                # We want dist from half-integer?
                # phase = (freqs / f0) % 1.0
                # center is 0.5.
                # dist = abs(phase - 0.5) * f0
                # But we only want 1.5, 2.5... not 0.5? "1.5 F0, 2.5 F0 etc."
                # Usually 0.5 F0 (sub-harmonic) is also possible but prompt says "1.5, 2.5".
                # Let's include 0.5 if it fits logic, but strictly prompt says "1.5 F0...".
                # Let's stick to n+0.5 where n >= 1.
                
                # Actually, iterate centers might be safer/cleaner.
                n_max = int(freqs[-1] / f0)
                for n in range(1, n_max + 1):
                    center = (n + 0.5) * f0
                    bw = 0.2 * f0
                    f_low = center - bw/2
                    f_high = center + bw/2
                    
                    mask_gap = (freqs >= f_low) & (freqs <= f_high)
                    hnr_gain_mask[mask_gap] = current_noise_gain
            
            
            # --- Apply All Gains ---
            total_gain = h1h2_gain_mask + slope_gain_mask + hnr_gain_mask
            
            # Update magnitude
            new_log_mag = curr_log_mag + total_gain
            
            # Convert back to linear
            new_mag = 10**(new_log_mag / 20.0)
            magnitude[:, i] = new_mag
            
        # Reconstruct
        # magnitude * exp(j*phase)
        Zxx_new = magnitude * np.exp(1j * phase)
        
        _, audio_rec = istft(Zxx_new, fs=self.fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        
        # Match length
        if len(audio_rec) > len(audio):
            audio_rec = audio_rec[:len(audio)]
        elif len(audio_rec) < len(audio):
            audio_rec = np.pad(audio_rec, (0, len(audio) - len(audio_rec)))
            
        return audio_rec

    def _find_peak(self, log_mag, freqs, target_f, search_ratio=0.1):
        """
        Find peak index near target_f.
        """
        # Define range
        delta = target_f * search_ratio
        f_min = target_f - delta
        f_max = target_f + delta
        
        # Indices
        mask = (freqs >= f_min) & (freqs <= f_max)
        if not np.any(mask):
            return None
            
        indices = np.where(mask)[0]
        # Find max in this region
        sub_mag = log_mag[indices]
        argmax = np.argmax(sub_mag)
        return indices[argmax]
