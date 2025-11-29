import argparse
import os
import sys
import tempfile
import subprocess
import shutil
import math

try:
    import numpy as np
except Exception:
    np = None
try:
    from scipy import signal as _scisig
except Exception:
    _scisig = None

def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p

class EpochTrackerPy:
    def __init__(self):
        self.external_frame_interval = 0.005
        self.internal_frame_interval = 0.002
        self.min_f0_search = 20.0
        self.max_f0_search = 500.0
        self.unvoiced_pulse_interval = 0.01
        self.unvoiced_cost = 0.9
        self.do_highpass = True
        self.do_hilbert_transform = False
        self.corner_frequency = 80.0
        self.filter_duration = 0.05
        self.frame_duration = 0.02
        self.lpc_frame_interval = 0.01
        self.preemphasis = 0.98
        self.noise_floor = 70.0
        self.peak_delay = 0.0004
        self.skew_delay = 0.00015
        self.peak_val_wt = 0.1
        self.peak_prominence_wt = 0.3
        self.peak_skew_wt = 0.1
        self.peak_quality_floor = 0.01
        self.time_span = 0.020
        self.level_change_den = 30.0
        self.min_rms_db = 20.0
        self.ref_dur = 0.02
        self.min_freq_for_rms = 100.0
        self.max_freq_for_rms = 1000.0
        self.rms_window_dur = 0.025
        self.correlation_dur = 0.0075
        self.correlation_thresh = 0.2
        self.reward = -1.5
        self.period_deviation_wt = 1.0
        self.peak_quality_wt = 1.3
        self.nccf_uv_peak_wt = 0.9
        self.period_wt = 0.0002
        self.level_wt = 0.8
        self.freq_trans_wt = 1.8
        self.voice_transition_factor = 1.4
        self.endpoint_padding = 0.01
        self.sample_rate = -1.0
        self.signal = []
        self.residual = []
        self.norm_residual = []
        self.peaks_debug = []
        self.bandpassed_rms = []
        self.voice_onset_prob = []
        self.voice_offset_prob = []
        self.prob_voiced = []
        self.best_corr = []
        self.window = []
        self.positive_rms = 0.0
        self.negative_rms = 0.0
        self.n_feature_frames = 0
        self.first_nccf_lag = 0
        self.n_nccf_lags = 0
        self.resid_peaks = []
        self.output = []

    def _round_up(self, v):
        return int(v + 0.5)

    def _hann_window(self, n):
        if (len(self.window) != n):
            self.window = [0.0] * n
            arg = 2.0 * math.pi / n
            for i in range(n):
                self.window[i] = 0.5 - (0.5 * math.cos((i + 0.5) * arg))

    def _make_linear_fir_half(self, fc, nf):
        if (nf % 2) != 1:
            nf = nf + 1
        n = (nf + 1) // 2
        twopi = 2.0 * math.pi
        coef = [0.0] * n
        coef[0] = 2.0 * fc
        c = math.pi
        fn = twopi * fc
        for i in range(1, n):
            coef[i] = math.sin(i * fn) / (c * i)
        fn = twopi / nf
        for i in range(n):
            coef[n - i - 1] *= (0.5 - (0.5 * math.cos(fn * (i + 0.5))))
        return coef, nf

    def _mirror_filter(self, half, nf, invert):
        filt = [0.0] * nf
        ncoefb2 = 1 + (nf // 2)
        integral = 0.0
        for t in range(ncoefb2 - 1):
            val = half[ncoefb2 - 1 - t]
            if not invert:
                filt[t] = val
                filt[nf - 1 - t] = val
            else:
                integral += val
                filt[t] = -val
                filt[nf - 1 - t] = -val
        if not invert:
            filt[nf - ncoefb2] = half[0]
        else:
            integral *= 2.0
            integral += half[0]
            filt[nf - ncoefb2] = integral - half[0]
        return np.array(filt, dtype=float) if np is not None else [float(x) for x in filt]

    def _highpass_filter(self, input_i16, sample_rate, corner_freq, filter_dur):
        nf = int(sample_rate * filter_dur)
        if (nf % 2) == 0:
            nf += 1
        fc = corner_freq / sample_rate
        half, nf = self._make_linear_fir_half(fc, nf)
        if np is None:
            raise RuntimeError('Python 实现需要 numpy 才能高效运行，请安装 numpy 或使用 --impl cpp')
        kernel = self._mirror_filter(half, nf, True)
        if _scisig is not None:
            y = _scisig.fftconvolve(input_i16.astype(float), kernel, mode='same')
        else:
            n = len(input_i16)
            L = _next_pow2(n + nf - 1)
            X = np.fft.rfft(input_i16.astype(float), L)
            H = np.fft.rfft(np.concatenate([kernel, np.zeros(L - nf)]), L)
            Y = X * H
            y = np.fft.irfft(Y, L)[:n]
        y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
        return y

    def _hilbert_transform(self, input_i16):
        if np is None:
            raise RuntimeError('Python 实现需要 numpy 才能高效运行，请安装 numpy 或使用 --impl cpp')
        n_input = len(input_i16)
        n_fft = _next_pow2(n_input)
        re = np.zeros(n_fft, dtype=float)
        im = np.zeros(n_fft, dtype=float)
        re[:n_input] = input_i16.astype(float)
        F = np.fft.fft(re + 1j * im)
        reF = F.real.copy()
        imF = F.imag.copy()
        for i in range(1, n_fft // 2):
            tmp = imF[i]
            imF[i] = -reF[i]
            reF[i] = tmp
        reF[0] = 0.0
        imF[0] = 0.0
        for i in range((n_fft // 2) + 1, n_fft):
            tmp = imF[i]
            imF[i] = reF[i]
            reF[i] = -tmp
        Fi = reF + 1j * imF
        x = np.fft.ifft(Fi).real
        return (x[:n_input] / n_fft).astype(np.float32)

    class _Lpc:
        @staticmethod
        def hann_window(din, n, preemp):
            w = [0.5 - 0.5 * math.cos(2.0 * math.pi * (i + 0.5) / n) for i in range(n)]
            dout = [0.0] * n
            if preemp != 0.0:
                for i in range(n):
                    dout[i] = w[i] * (din[i + 1] - preemp * din[i])
            else:
                for i in range(n):
                    dout[i] = w[i] * din[i]
            return dout
        @staticmethod
        def autoc(windowsize, s, p):
            sum0 = 0.0
            for i in range(windowsize):
                sum0 += s[i] * s[i]
            if sum0 == 0.0:
                r = [0.0] * (p + 1)
                r[0] = 1.0
                e = 1.0
                return r, e
            e = math.sqrt(sum0 / windowsize)
            r = [0.0] * (p + 1)
            r[0] = 1.0
            sum0_inv = 1.0 / sum0
            for lag in range(1, p + 1):
                acc = 0.0
                for j in range(windowsize - lag):
                    acc += s[j] * s[j + lag]
                r[lag] = sum0_inv * acc
            return r, e
        @staticmethod
        def durbin(r, p):
            k = [0.0] * p
            a = [0.0] * (p + 1)
            b = [0.0] * (p + 1)
            e = r[0]
            k[0] = -r[1] / e
            a[0] = k[0]
            e *= (1.0 - k[0] * k[0])
            for i in range(1, p):
                s = 0.0
                for j in range(0, i):
                    s -= a[j] * r[i - j]
                k[i] = (s - r[i + 1]) / e
                a[i] = k[i]
                for j in range(0, i + 1):
                    b[j] = a[j]
                for j in range(0, i):
                    a[j] += k[i] * b[i - j - 1]
                e *= (1.0 - k[i] * k[i])
            return k, a, e
        @staticmethod
        def compute_lpc(lpc_ord, noise_floor, wsize, data, preemp):
            dwind = EpochTrackerPy._Lpc.hann_window(data, wsize, preemp)
            r, en = EpochTrackerPy._Lpc.autoc(wsize, dwind, lpc_ord)
            if noise_floor > 1.0:
                ffact = 1.0 / (1.0 + math.exp((-noise_floor / 20.0) * math.log(10.0)))
                rho = r[:]
                for i in range(1, lpc_ord + 1):
                    rho[i] = ffact * r[i]
                rho[0] = r[0]
                r = rho
            k, a_sub, er = EpochTrackerPy._Lpc.durbin(r, lpc_ord)
            a = [0.0] * (lpc_ord + 1)
            a[0] = 1.0
            for i in range(lpc_ord):
                a[i + 1] = a_sub[i]
            wfact = 0.612372
            rms = en / wfact
            return a, er, rms

    def _lpc_dc_gain(self, lpc):
        s = 0.0
        for v in lpc:
            s += v
        return s if s > 0.0 else 1.0

    def _make_deltas(self, now, nxt, n_steps):
        return (nxt - now) / float(n_steps)

    def init(self, input_i16, sample_rate, min_f0, max_f0, do_highpass, do_hilbert):
        if (sample_rate <= 6000.0) or (len(input_i16) <= int(sample_rate * 0.05)) or (min_f0 >= max_f0) or (min_f0 <= 0.0):
            return False
        self.min_f0_search = min_f0
        self.max_f0_search = max_f0
        self.sample_rate = float(sample_rate)
        p = input_i16
        if do_highpass:
            p = self._highpass_filter(p, self.sample_rate, self.corner_frequency, self.filter_duration)
        self.signal = [0.0] * len(p)
        if do_hilbert:
            h = self._hilbert_transform(p)
            for i in range(len(p)):
                self.signal[i] = float(h[i])
        else:
            for i in range(len(p)):
                self.signal[i] = float(p[i])
        return True

    def window_fn(self, input_data, offset, size):
        if np is None:
            raise RuntimeError('Python 实现需要 numpy 才能高效运行，请安装 numpy 或使用 --impl cpp')
        self._hann_window(size)
        data = np.array(input_data[offset:offset+size], dtype=float)
        win = np.array(self.window, dtype=float)
        return data * win

    def get_bandpassed_rms(self, input_data, low_limit, high_limit, frame_interval, frame_dur):
        sr = self.sample_rate
        frame_step = self._round_up(sr * frame_interval)
        frame_size = self._round_up(sr * frame_dur)
        n_frames = 1 + ((len(input_data) - frame_size) // frame_step)
        if n_frames < 2:
            return False
        out = [0.0] * n_frames
        fft_size = _next_pow2(frame_size)
        first_bin = self._round_up(fft_size * low_limit / sr)
        last_bin = self._round_up(fft_size * high_limit / sr)
        first_frame = frame_size // (2 * frame_step)
        if (first_frame * 2 * frame_step) < frame_size:
            first_frame += 1
        for frame in range(first_frame, n_frames):
            win = self.window_fn(input_data, (frame - first_frame) * frame_step, frame_size)
            if np is None:
                raise RuntimeError('Python 实现需要 numpy 才能高效运行，请安装 numpy 或使用 --impl cpp')
            re = np.zeros(fft_size, dtype=float)
            re[:frame_size] = win
            F = np.fft.fft(re)
            mag2 = F.real * F.real + F.imag * F.imag
            band_sum = float(np.sum(mag2[first_bin:last_bin+1]))
            rms = 20.0 * math.log10(1.0 + math.sqrt(band_sum / float(max(1, (last_bin - first_bin + 1)))))
            out[frame] = rms
            if frame == first_frame:
                for b in range(0, first_frame):
                    out[b] = rms
        self.bandpassed_rms = out
        return True

    def get_symmetry_stats(self, data):
        n_input = len(data)
        s = float(sum(data))
        mean = s / n_input
        p_sum = 0.0
        n_sum = 0.0
        n_p = 0
        n_n = 0
        for v in data:
            val = v - mean
            if val > 0.0:
                p_sum += val * val
                n_p += 1
            elif val < 0.0:
                n_sum += val * val
                n_n += 1
        pos_rms = math.sqrt(p_sum / max(n_p, 1))
        neg_rms = math.sqrt(n_sum / max(n_n, 1))
        return pos_rms, neg_rms, mean

    def normalize_amplitude(self, input_data):
        n_input = len(input_data)
        ref_size = self._round_up(self.sample_rate * self.ref_dur)
        out = [0.0] * n_input
        self._hann_window(ref_size)
        ref_by_2 = ref_size // 2
        frame_step = self._round_up(self.sample_rate * self.internal_frame_interval)
        limit = n_input - ref_size
        frame_limit = ref_by_2
        data_p = 0
        frame_p = 0
        old_inv_rms = 0.0
        while frame_p < limit:
            ref_energy = 1.0
            for i in range(ref_size):
                val = self.window[i] * input_data[i + frame_p]
                ref_energy += val * val
            inv_rms = math.sqrt(float(ref_size) / ref_energy)
            delta_inv_rms = 0.0
            if frame_p > 0:
                delta_inv_rms = (inv_rms - old_inv_rms) / frame_step
            else:
                old_inv_rms = inv_rms
            for i in range(frame_limit):
                out[data_p] = input_data[data_p] * old_inv_rms
                old_inv_rms += delta_inv_rms
                data_p += 1
            frame_limit = frame_step
            frame_p += frame_step
        while data_p < n_input:
            out[data_p] = input_data[data_p] * old_inv_rms
            data_p += 1
        self.norm_residual = out

    def compute_features(self):
        if self.sample_rate <= 0.0:
            return False
        if not self.get_bandpassed_rms(self.signal, self.min_freq_for_rms, self.max_freq_for_rms, self.internal_frame_interval, self.rms_window_dur):
            return False
        if not self.get_lpc_residual():
            return False
        self.n_feature_frames = len(self.bandpassed_rms)
        pos_rms, neg_rms, mean = self.get_symmetry_stats(self.residual)
        self.positive_rms = pos_rms
        self.negative_rms = neg_rms
        if self.positive_rms > self.negative_rms:
            for i in range(len(self.residual)):
                self.residual[i] = -self.residual[i]
                self.signal[i] = -self.signal[i]
        self.normalize_amplitude(self.residual)
        self.get_residual_pulses()
        self.get_pulse_correlations(self.correlation_dur, self.correlation_thresh)
        self.get_voice_transition_features()
        self.get_rms_voicing_modulator()
        return True

    def get_lpc_residual(self):
        input_f = [float(x) for x in self.signal]
        n_input = len(input_f)
        if n_input <= 0:
            return False
        out = [0.0] * n_input
        frame_step = self._round_up(self.sample_rate * self.lpc_frame_interval)
        frame_size = self._round_up(self.sample_rate * self.frame_duration)
        n_frames = 1 + ((n_input - frame_size) // frame_step)
        n_analyzed = ((n_frames - 1) * frame_step) + frame_size
        if n_analyzed <= n_input:
            n_frames -= 1
            if n_frames <= 0:
                return False
        order = int(2.5 + (self.sample_rate / 1000.0))
        lpc, norm_error, preemp_rms = self._Lpc.compute_lpc(order, self.noise_floor, frame_size, input_f[:frame_size+1], self.preemphasis)
        old_lpc = lpc[:]
        delta_lpc = [0.0] * (order + 1)
        for i in range(order + 1):
            delta_lpc[i] = 0.0
            out[i] = 0.0
        old_gain = self._lpc_dc_gain(old_lpc)
        n_to_filter = (frame_size // 2) - order
        input_p = 0
        output_p = order
        proc_p = 0
        for _ in range(n_frames):
            lpc, norm_error, preemp_rms = self._Lpc.compute_lpc(order, self.noise_floor, frame_size, input_f[input_p:input_p+frame_size+1], self.preemphasis)
            new_gain = self._lpc_dc_gain(lpc)
            delta_gain = (new_gain - old_gain) / float(n_to_filter)
            delta_lpc = [(lpc[i] - old_lpc[i]) / float(n_to_filter) for i in range(order + 1)]
            for sample in range(n_to_filter):
                sumv = 0.0
                mem = proc_p
                for k in range(order, 0, -1):
                    sumv += old_lpc[k] * input_f[mem]
                    old_lpc[k] += delta_lpc[k]
                    mem += 1
                sumv += input_f[mem]
                out[output_p] = sumv / old_gain
                old_gain += delta_gain
                proc_p += 1
                output_p += 1
            input_p += frame_step
            n_to_filter = frame_step
        self.residual = out
        return True

    def get_residual_pulses(self):
        peak_ind = self._round_up(self.peak_delay * self.sample_rate)
        skew_ind = self._round_up(self.skew_delay * self.sample_rate)
        min_peak = -1.0
        limit = len(self.norm_residual) - peak_ind
        self.resid_peaks = []
        self.peaks_debug = [0.0] * len(self.residual)
        for i in range(peak_ind, limit):
            val = self.norm_residual[i]
            if val > min_peak:
                continue
            if (self.norm_residual[i-1] > val) and (val <= self.norm_residual[i+1]):
                vm_peak = self.norm_residual[i - peak_ind]
                vp_peak = self.norm_residual[i + peak_ind]
                if (vm_peak < val) or (vp_peak < val):
                    continue
                vm_skew = self.norm_residual[i - skew_ind]
                vp_skew = self.norm_residual[i + skew_ind]
                sharp = (0.5 * (vp_peak + vm_peak)) - val
                skew = -(vm_skew - vp_skew)
                p = {
                    'resid_index': i,
                    'frame_index': self._round_up((i / self.sample_rate) / self.internal_frame_interval),
                    'peak_quality': (-val * self.peak_val_wt) + (skew * self.peak_skew_wt) + (sharp * self.peak_prominence_wt),
                    'nccf': [],
                    'nccf_periods': [],
                    'future': [],
                    'past': []
                }
                if p['frame_index'] >= self.n_feature_frames:
                    p['frame_index'] = self.n_feature_frames - 1
                if p['peak_quality'] < self.peak_quality_floor:
                    p['peak_quality'] = self.peak_quality_floor
                self.resid_peaks.append(p)
                self.peaks_debug[i] = p['peak_quality']

    def _find_nccf_peaks(self, arr, thresh):
        limit = len(arr) - 1
        out = []
        max_val = 0.0
        max_index = 1
        max_out_index = 0
        for i in range(1, limit):
            val = arr[i]
            if (val > thresh) and (val > arr[i-1]) and (val >= arr[i+1]):
                if val > max_val:
                    max_val = val
                    max_out_index = len(out)
                    max_index = i
                out.append(i)
        if (len(out) > 1) and (max_out_index > 0):
            hold = out[0]
            out[0] = out[max_out_index]
            out[max_out_index] = hold
        else:
            if len(out) <= 0:
                out.append(max_index)
        return out

    def _cross_correlation(self, data, start, first_lag, n_lags, size):
        if np is None:
            raise RuntimeError('Python 实现需要 numpy 才能高效运行，请安装 numpy 或使用 --impl cpp')
        input_seg = np.array(data[start:start + size + first_lag + n_lags], dtype=float)
        energy = float(np.dot(input_seg[:size], input_seg[:size]))
        corr = [0.0] * n_lags
        if energy == 0.0:
            return corr
        lag_energy = float(np.dot(input_seg[first_lag:first_lag+size], input_seg[first_lag:first_lag+size]))
        last_lag = first_lag + n_lags
        oind = 0
        for lag in range(first_lag, last_lag):
            sumv = float(np.dot(input_seg[:size], input_seg[lag:lag+size]))
            if lag_energy <= 0.0:
                lag_energy = 1.0
            corr[oind] = sumv / math.sqrt(energy * lag_energy)
            lag_energy -= input_seg[lag] * input_seg[lag]
            lag_energy += input_seg[lag + size] * input_seg[lag + size]
            oind += 1
        return corr

    def get_pulse_correlations(self, window_dur, peak_thresh):
        self.first_nccf_lag = self._round_up(self.sample_rate / self.max_f0_search)
        max_lag = self._round_up(self.sample_rate / self.min_f0_search)
        self.n_nccf_lags = max_lag - self.first_nccf_lag
        window_size = self._round_up(window_dur * self.sample_rate)
        half_wind = window_size // 2
        frame_size = window_size + max_lag
        mixture = [(0.7 * self.residual[i]) + (0.3 * self.signal[i]) for i in range(len(self.residual))]
        min_step = self._round_up(self.sample_rate * 0.001)
        old_start = - (2 * min_step)
        for idx in range(len(self.resid_peaks)):
            start = self.resid_peaks[idx]['resid_index'] - half_wind
            if start < 0:
                start = 0
            end = start + frame_size
            if (end >= len(mixture)) or ((start - old_start) < min_step):
                self.resid_peaks[idx]['nccf'] = self.resid_peaks[idx-1]['nccf'] if idx > 0 else [0.0] * self.n_nccf_lags
                self.resid_peaks[idx]['nccf_periods'] = self.resid_peaks[idx-1]['nccf_periods'] if idx > 0 else [self.first_nccf_lag]
            else:
                corr = self._cross_correlation(mixture, start, self.first_nccf_lag, self.n_nccf_lags, window_size)
                self.resid_peaks[idx]['nccf'] = corr
                peaks = self._find_nccf_peaks(corr, peak_thresh)
                self.resid_peaks[idx]['nccf_periods'] = [p + self.first_nccf_lag for p in peaks]
                old_start = start

    def get_voice_transition_features(self):
        frame_offset = self._round_up(0.5 * self.time_span / self.internal_frame_interval)
        if frame_offset <= 0:
            frame_offset = 1
        self.voice_onset_prob = [0.0] * self.n_feature_frames
        self.voice_offset_prob = [0.0] * self.n_feature_frames
        limit = self.n_feature_frames - frame_offset
        for frame in range(frame_offset, limit):
            delta_rms = (self.bandpassed_rms[frame + frame_offset] - self.bandpassed_rms[frame - frame_offset]) / self.level_change_den
            if delta_rms > 1.0:
                delta_rms = 1.0
            elif delta_rms < -1.0:
                delta_rms = -1.0
            prob_onset = delta_rms
            prob_offset = -prob_onset
            if prob_onset > 1.0:
                prob_onset = 1.0
            elif prob_onset < 0.0:
                prob_onset = 0.0
            if prob_offset > 1.0:
                prob_offset = 1.0
            elif prob_offset < 0.0:
                prob_offset = 0.0
            self.voice_onset_prob[frame] = prob_onset
            self.voice_offset_prob[frame] = prob_offset
        for frame in range(0, frame_offset):
            bframe = self.n_feature_frames - 1 - frame
            self.voice_onset_prob[frame] = 0.0
            self.voice_offset_prob[frame] = 0.0
            self.voice_onset_prob[bframe] = 0.0
            self.voice_offset_prob[bframe] = 0.0

    def get_rms_voicing_modulator(self):
        if not self.bandpassed_rms:
            self.prob_voiced = []
            return
        min_val = self.bandpassed_rms[0]
        max_val = min_val
        for v in self.bandpassed_rms[1:]:
            if v < min_val:
                min_val = v
            elif v > max_val:
                max_val = v
        if min_val < self.min_rms_db:
            min_val = self.min_rms_db
        rng = max_val - min_val
        if rng < 1.0:
            rng = 1.0
        self.prob_voiced = [0.0] * len(self.bandpassed_rms)
        for i in range(len(self.bandpassed_rms)):
            v = (self.bandpassed_rms[i] - min_val) / rng
            if v < 0.0:
                v = 0.0
            self.prob_voiced[i] = v

    def create_period_lattice(self):
        low_period = self._round_up(self.sample_rate / self.max_f0_search)
        high_period = self._round_up(self.sample_rate / self.min_f0_search)
        self.best_corr = []
        for peak_idx in range(len(self.resid_peaks)):
            frame_index = self.resid_peaks[peak_idx]['frame_index']
            resid_index = self.resid_peaks[peak_idx]['resid_index']
            min_period = resid_index + low_period
            max_period = resid_index + high_period
            time = resid_index / self.sample_rate
            best_nccf_period = self.resid_peaks[peak_idx]['nccf_periods'][0]
            best_cc_val = self.resid_peaks[peak_idx]['nccf'][best_nccf_period - self.first_nccf_lag]
            self.best_corr.append(time)
            self.best_corr.append(best_cc_val)
            uv_cand = {'voiced': False, 'start_peak': peak_idx, 'cost_sum': 0.0, 'local_cost': 0.0, 'best_prev_cand': -1}
            next_cands_created = 0
            for npeak in range(peak_idx + 1, len(self.resid_peaks)):
                iperiod = self.resid_peaks[npeak]['resid_index'] - resid_index
                if self.resid_peaks[npeak]['resid_index'] >= min_period:
                    fperiod = float(iperiod)
                    cc_peak = 0
                    min_period_diff = abs(math.log(fperiod / best_nccf_period))
                    for cc_peak_ind in range(1, len(self.resid_peaks[peak_idx]['nccf_periods'])):
                        nccf_period = self.resid_peaks[peak_idx]['nccf_periods'][cc_peak_ind]
                        test_diff = abs(math.log(fperiod / nccf_period))
                        if test_diff < min_period_diff:
                            min_period_diff = test_diff
                            cc_peak = cc_peak_ind
                    v_cand = {'voiced': True, 'period': iperiod}
                    cc_index = iperiod - self.first_nccf_lag
                    if (cc_index >= 0) and (cc_index < self.n_nccf_lags):
                        cc_value = self.resid_peaks[peak_idx]['nccf'][cc_index]
                    else:
                        peak_cc_index = self.resid_peaks[peak_idx]['nccf_periods'][cc_peak] - self.first_nccf_lag
                        cc_value = self.resid_peaks[peak_idx]['nccf'][peak_cc_index]
                    per_dev_cost = self.period_deviation_wt * min_period_diff
                    level_cost = self.level_wt * (1.0 - self.prob_voiced[frame_index])
                    period_cost = fperiod * self.period_wt
                    peak_qual_cost = self.peak_quality_wt / (self.resid_peaks[npeak]['peak_quality'] + self.resid_peaks[peak_idx]['peak_quality'])
                    local_cost = (1.0 - cc_value) + per_dev_cost + peak_qual_cost + level_cost + period_cost + self.reward
                    v_cand['local_cost'] = local_cost
                    if local_cost < 1.0e30:
                        uv_cand['period'] = iperiod
                        level_cost_uv = self.level_wt * self.prob_voiced[frame_index]
                        uv_cand['local_cost'] = (self.nccf_uv_peak_wt * cc_value) + level_cost_uv + self.unvoiced_cost + self.reward
                        uv_cand['end_peak'] = npeak
                        uv_cand['closest_nccf_period'] = self.resid_peaks[peak_idx]['nccf_periods'][cc_peak]
                    v_cand['start_peak'] = peak_idx
                    v_cand['end_peak'] = npeak
                    v_cand['closest_nccf_period'] = self.resid_peaks[peak_idx]['nccf_periods'][cc_peak]
                    v_cand['cost_sum'] = 0.0
                    v_cand['best_prev_cand'] = -1
                    self.resid_peaks[peak_idx]['future'].append(v_cand)
                    self.resid_peaks[npeak]['past'].append(v_cand)
                    next_cands_created += 1
                    if self.resid_peaks[npeak]['resid_index'] >= max_period:
                        break
            if next_cands_created:
                self.resid_peaks[peak_idx]['future'].append(uv_cand)
                self.resid_peaks[uv_cand['end_peak']]['past'].append(uv_cand)
            else:
                pass
            if len(self.resid_peaks[peak_idx]['past']) == 0:
                for pp in range(len(self.resid_peaks[peak_idx]['future'])):
                    self.resid_peaks[peak_idx]['future'][pp]['cost_sum'] = self.resid_peaks[peak_idx]['future'][pp]['local_cost']
                    self.resid_peaks[peak_idx]['future'][pp]['best_prev_cand'] = -1
            else:
                uv_hyps_found = 0
                lowest_cost = self.resid_peaks[peak_idx]['past'][0]['local_cost']
                lowest_index = 0
                for pcand in range(len(self.resid_peaks[peak_idx]['past'])):
                    if not self.resid_peaks[peak_idx]['past'][pcand]['voiced']:
                        uv_hyps_found += 1
                    else:
                        if self.resid_peaks[peak_idx]['past'][pcand]['local_cost'] < lowest_cost:
                            lowest_index = pcand
                            lowest_cost = self.resid_peaks[peak_idx]['past'][pcand]['local_cost']
                if not uv_hyps_found:
                    start_peak = self.resid_peaks[peak_idx]['past'][lowest_index]['start_peak']
                    uv = {'voiced': False}
                    uv['start_peak'] = start_peak
                    uv['end_peak'] = peak_idx
                    uv['period'] = self.resid_peaks[peak_idx]['past'][lowest_index]['period']
                    uv['closest_nccf_period'] = self.resid_peaks[peak_idx]['past'][lowest_index]['closest_nccf_period']
                    uv['cost_sum'] = 0.0
                    uv['local_cost'] = 0.0
                    uv['best_prev_cand'] = -1
                    llevel_cost = self.level_wt * self.prob_voiced[self.resid_peaks[start_peak]['frame_index']]
                    lcc_index = uv['period'] - self.first_nccf_lag
                    if (lcc_index >= 0) and (lcc_index < self.n_nccf_lags):
                        lcc_value = self.resid_peaks[start_peak]['nccf'][lcc_index]
                    else:
                        peak_cc_index = uv['closest_nccf_period'] - self.first_nccf_lag
                        lcc_value = self.resid_peaks[start_peak]['nccf'][peak_cc_index]
                    uv['local_cost'] = (self.nccf_uv_peak_wt * lcc_value) + llevel_cost + self.unvoiced_cost + self.reward
                    self.resid_peaks[start_peak]['future'].append(uv)
                    self.resid_peaks[peak_idx]['past'].append(uv)

    def do_dynamic_programming(self):
        for peak in range(len(self.resid_peaks)):
            if len(self.resid_peaks[peak]['past']) == 0:
                continue
            for fhyp in range(len(self.resid_peaks[peak]['future'])):
                min_cost = 1.0e30
                min_index = 0
                forward_period = float(self.resid_peaks[peak]['future'][fhyp]['period']) if 'period' in self.resid_peaks[peak]['future'][fhyp] else 0.0
                for phyp in range(len(self.resid_peaks[peak]['past'])):
                    sum_cost = 0.0
                    if self.resid_peaks[peak]['future'][fhyp].get('voiced', False) and self.resid_peaks[peak]['past'][phyp].get('voiced', False):
                        f_trans_cost = self.freq_trans_wt * abs(math.log(forward_period / float(self.resid_peaks[peak]['past'][phyp]['period'])))
                        sum_cost = f_trans_cost + self.resid_peaks[peak]['past'][phyp]['cost_sum']
                    else:
                        if self.resid_peaks[peak]['future'][fhyp].get('voiced', False) and (not self.resid_peaks[peak]['past'][phyp].get('voiced', False)):
                            v_transition_cost = self.voice_transition_factor * (1.0 - self.voice_onset_prob[self.resid_peaks[peak]['frame_index']])
                            sum_cost = self.resid_peaks[peak]['past'][phyp]['cost_sum'] + v_transition_cost
                        else:
                            if (not self.resid_peaks[peak]['future'][fhyp].get('voiced', False)) and self.resid_peaks[peak]['past'][phyp].get('voiced', False):
                                v_transition_cost = self.voice_transition_factor * (1.0 - self.voice_offset_prob[self.resid_peaks[peak]['frame_index']])
                                sum_cost = self.resid_peaks[peak]['past'][phyp]['cost_sum'] + v_transition_cost
                            else:
                                sum_cost = self.resid_peaks[peak]['past'][phyp]['cost_sum']
                    if sum_cost < min_cost:
                        min_cost = sum_cost
                        min_index = phyp
                self.resid_peaks[peak]['future'][fhyp]['cost_sum'] = self.resid_peaks[peak]['future'][fhyp]['local_cost'] + min_cost
                self.resid_peaks[peak]['future'][fhyp]['best_prev_cand'] = min_index

    def backtrack_and_save_output(self):
        if len(self.resid_peaks) == 0:
            return False
        min_cost = 1.0e30
        min_index = 0
        end = 0
        for peak in range(len(self.resid_peaks) - 1, 0, -1):
            if len(self.resid_peaks[peak]['past']) > 1:
                for ind in range(len(self.resid_peaks[peak]['past'])):
                    if self.resid_peaks[peak]['past'][ind]['cost_sum'] < min_cost:
                        min_cost = self.resid_peaks[peak]['past'][ind]['cost_sum']
                        min_index = ind
                end = peak
                break
        if end == 0:
            return False
        self.output = []
        while True:
            start_peak = self.resid_peaks[end]['past'][min_index]['start_peak']
            tr = {}
            tr['resid_index'] = self.resid_peaks[start_peak]['resid_index']
            if self.resid_peaks[end]['past'][min_index]['voiced']:
                nccf_period = self.resid_peaks[end]['past'][min_index]['closest_nccf_period']
                tr['f0'] = self.sample_rate / float(nccf_period)
                tr['voiced'] = True
            else:
                tr['f0'] = 0.0
                tr['voiced'] = False
            cc_index = self.resid_peaks[end]['past'][min_index]['period'] - self.first_nccf_lag
            if (cc_index >= 0) and (cc_index < self.n_nccf_lags):
                tr['nccf_value'] = self.resid_peaks[start_peak]['nccf'][cc_index]
            else:
                peak_cc_index = self.resid_peaks[end]['past'][min_index]['closest_nccf_period'] - self.first_nccf_lag
                tr['nccf_value'] = self.resid_peaks[start_peak]['nccf'][peak_cc_index]
            self.output.append(tr)
            new_end = self.resid_peaks[end]['past'][min_index]['start_peak']
            min_index = self.resid_peaks[end]['past'][min_index]['best_prev_cand']
            if min_index < 0:
                break
            end = new_end
        return True

    def track_epochs(self):
        self.create_period_lattice()
        self.do_dynamic_programming()
        return self.backtrack_and_save_output()

    def get_filled_epochs(self, unvoiced_pm_interval):
        times = []
        voicing = []
        final_time = len(self.norm_residual) / self.sample_rate
        limit = len(self.output) - 1
        i = limit
        while i >= 0:
            i_old = i
            time = self.output[i]['resid_index'] / self.sample_rate
            if self.output[i]['voiced'] or ((i < limit) and (self.output[i+1]['voiced'])):
                times.append(time)
                voicing.append(1)
                i -= 1
            if i == limit:
                time = 0.0
            if (i > 0) and (not self.output[i]['voiced']) and (time < final_time):
                while i > 0:
                    if self.output[i]['voiced']:
                        break
                    i -= 1
                next_time = final_time
                fill_ind = 1
                if i > 0:
                    next_time = (self.output[i]['resid_index'] / self.sample_rate) - (1.0 / self.max_f0_search)
                now = time + (fill_ind * unvoiced_pm_interval)
                while now < next_time:
                    times.append(now)
                    voicing.append(0)
                    fill_ind += 1
                    now = time + (fill_ind * unvoiced_pm_interval)
            if i == i_old:
                i -= 1
        return times, voicing

    def resample_and_return_results(self, resample_interval):
        if (self.sample_rate <= 0.0) or (len(self.output) == 0):
            return None, None
        if resample_interval <= 0.0:
            return None, None
        last_time = (self.output[0]['resid_index'] / self.sample_rate) + self.endpoint_padding
        n_frames = self._round_up(last_time / resample_interval)
        f0 = [0.0] * n_frames
        corr = [0.0] * n_frames
        limit = len(self.output) - 1
        prev_frame = 0
        prev_f0 = self.output[limit]['f0']
        prev_corr = self.output[limit]['nccf_value']
        for i in range(limit, -1, -1):
            frame = self._round_up(self.output[i]['resid_index'] / (self.sample_rate * resample_interval))
            f0[frame] = self.output[i]['f0']
            corr[frame] = self.output[i]['nccf_value']
            if (frame - prev_frame) > 1:
                for fr in range(prev_frame + 1, frame):
                    f0[fr] = prev_f0
                    corr[fr] = prev_corr
            prev_frame = frame
            prev_corr = self.output[i]['nccf_value']
            prev_f0 = self.output[i]['f0']
        for frame in range(prev_frame, n_frames):
            f0[frame] = prev_f0
            corr[frame] = prev_corr
        return f0, corr

def parse_est_track(path):
    times=[]
    voiced=[]
    values=[]
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        in_header=True
        for line in f:
            s=line.strip()
            if not s:
                continue
            if in_header:
                if s=="EST_Header_End":
                    in_header=False
                continue
            parts=s.split()
            if len(parts)<3:
                continue
            try:
                t=float(parts[0])
                v=int(parts[1])
                val=float(parts[2])
            except:
                continue
            times.append(t)
            voiced.append(v)
            values.append(val)
    return times,voiced,values

def find_reaper(explicit=None):
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        p=shutil.which(explicit)
        if p:
            return p
    p=shutil.which('reaper')
    if p:
        return p
    p=shutil.which('reaper.exe')
    if p:
        return p
    base=os.path.dirname(os.path.abspath(__file__))
    candidates=[
        os.path.join(base,'build','Release','reaper.exe'),
        os.path.join(base,'build','reaper.exe'),
        os.path.join(base,'Release','reaper.exe'),
        os.path.join(base,'reaper.exe'),
        os.path.join(base,'reaper')
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise RuntimeError('reaper executable not found')

def run_reaper(reaper_bin,input_wav,f0_out,frame_interval,min_f0,max_f0,hilbert,no_highpass):
    cmd=[reaper_bin,'-i',input_wav,'-f',f0_out,'-a','-e',str(frame_interval),'-m',str(min_f0),'-x',str(max_f0)]
    if hilbert:
        cmd.append('-t')
    if no_highpass:
        cmd.append('-s')
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if r.returncode!=0:
        raise RuntimeError(r.stderr.decode(errors='ignore') or 'reaper failed')

def run_python_impl(input_wav,frame_interval,min_f0,max_f0,hilbert,no_highpass):
    import wave
    with wave.open(input_wav,'rb') as w:
        ch=w.getnchannels()
        sr=w.getframerate()
        sw=w.getsampwidth()
        n=w.getnframes()
        if ch!=1 or sw!=2:
            raise RuntimeError('only mono 16-bit PCM wav supported')
        buf=w.readframes(n)
    if np is None:
        raise RuntimeError('Python 实现需要 numpy 才能高效运行，请安装 numpy 或使用 --impl cpp')
    data=np.frombuffer(buf,dtype=np.int16).copy()
    et=EpochTrackerPy()
    ok=et.init(data,float(sr),float(min_f0),float(max_f0),not no_highpass,bool(hilbert))
    if not ok:
        raise RuntimeError('init failed')
    if not et.compute_features():
        raise RuntimeError('compute_features failed')
    if not et.track_epochs():
        raise RuntimeError('track_epochs failed')
    f0,corr=et.resample_and_return_results(float(frame_interval))
    if f0 is None:
        raise RuntimeError('resample failed')
    times=[i*float(frame_interval) for i in range(len(f0))]
    return times,[1 if v>0.0 else 0 for v in f0],f0

def plot_f0(times,values,plot_out,show):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        t=np.array(times,dtype=float)
        v=np.array(values,dtype=float)
        v=np.where(v>0.0,v,np.nan)
        fig=plt.figure(figsize=(10,4))
        ax=fig.add_subplot(111)
        ax.plot(t,v,linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('F0 (Hz)')
        ax.set_title('REAPER F0')
        ax.grid(True,alpha=0.3)
        if plot_out:
            fig.savefig(plot_out,bbox_inches='tight',dpi=150)
        if show:
            plt.show()
        return
    except Exception:
        pass
    try:
        from PIL import Image, ImageDraw
        w,h=1200,480
        img=Image.new('RGB',(w,h),(255,255,255))
        dr=ImageDraw.Draw(img)
        if not times:
            raise RuntimeError('empty F0 output')
        tmin=min(times)
        tmax=max(times)
        fvals=[v for v in values if v>0]
        if not fvals:
            fmin=0.0
            fmax=1.0
        else:
            fmin=min(fvals)
            fmax=max(fvals)
        def xmap(t):
            return int(40+(w-80)*(0 if tmax==tmin else (t-tmin)/(tmax-tmin)))
        def ymap(f):
            if f<=0:
                return None
            y= int(40+(h-80)*(1.0-(f-fmin)/(fmax-fmin if fmax>fmin else 1)))
            return max(40,min(h-40,y))
        last=None
        for t,f in zip(times,values):
            x=xmap(t)
            y=ymap(f)
            if y is None:
                last=None
                continue
            if last is not None:
                dr.line([last,(x,y)],fill=(0,100,240),width=2)
            last=(x,y)
        dr.rectangle([40,40,w-40,h-40],outline=(0,0,0))
        if plot_out:
            img.save(plot_out)
        if show:
            img.show()
        return
    except Exception:
        pass
    if not times:
        raise RuntimeError('empty F0 output')
    w,h=1200,480
    tmin=min(times)
    tmax=max(times)
    fvals=[v for v in values if v>0]
    if not fvals:
        fmin=0.0
        fmax=1.0
    else:
        fmin=min(fvals)
        fmax=max(fvals)
    def xmap(t):
        return 40+(w-80)*(0 if tmax==tmin else (t-tmin)/(tmax-tmin))
    def ymap(f):
        if f<=0:
            return None
        y= 40+(h-80)*(1.0-(f-fmin)/(fmax-fmin if fmax>fmin else 1))
        return max(40,min(h-40,y))
    pts=[]
    for t,f in zip(times,values):
        y=ymap(f)
        if y is None:
            if pts:
                pts.append('M')
            continue
        x=xmap(t)
        pts.append((x,y))
    svg_lines=[]
    pen_up=True
    for it in pts:
        if it=='M':
            pen_up=True
            continue
        x,y=it
        if pen_up:
            svg_lines.append(f"M {x:.2f},{y:.2f}")
            pen_up=False
        else:
            svg_lines.append(f"L {x:.2f},{y:.2f}")
    base,ext=os.path.splitext(plot_out or 'f0.svg')
    svg_out=base+'.svg'
    with open(svg_out,'w',encoding='utf-8') as f:
        f.write(f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>")
        f.write(f"<rect x='40' y='40' width='{w-80}' height='{h-80}' fill='white' stroke='black'/>")
        path_data=' '.join(svg_lines)
        f.write(f"<path d='{path_data}' stroke='rgb(0,100,240)' fill='none' stroke-width='2'/>")
        f.write("</svg>")

def write_csv(times, values, csv_out):
    import csv
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['time', 'f0'])
        for t, v in zip(times, values):
            w.writerow([f"{t:.6f}", ("" if v <= 0.0 else f"{v:.6f}")])

def process_file(impl,reaper_bin,input_wav,frame_interval,min_f0,max_f0,hilbert,no_highpass,plot_out,show,csv_out,csv_only):
    if impl=='cpp':
        with tempfile.TemporaryDirectory() as td:
            f0_out=os.path.join(td,'out.f0')
            run_reaper(reaper_bin,input_wav,f0_out,frame_interval,min_f0,max_f0,hilbert,no_highpass)
            times,voiced,values=parse_est_track(f0_out)
    else:
        times,voiced,values=run_python_impl(input_wav,frame_interval,min_f0,max_f0,hilbert,no_highpass)
    if not times:
        raise RuntimeError('empty F0 output')
    if csv_out:
        write_csv(times, values, csv_out)
    if not csv_only:
        plot_f0(times,values,plot_out,show)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('input_wav', nargs='?')
    p.add_argument('--reaper-bin',default=None)
    p.add_argument('--impl',choices=['cpp','py'],default='cpp')
    p.add_argument('--frame-interval',type=float,default=0.005)
    p.add_argument('--min-f0',type=float,default=40.0)
    p.add_argument('--max-f0',type=float,default=500.0)
    p.add_argument('--hilbert',action='store_true')
    p.add_argument('--no-highpass',action='store_true')
    p.add_argument('--plot-out',default=None)
    p.add_argument('--csv-out',default=None)
    p.add_argument('--csv-only',action='store_true')
    p.add_argument('--out-dir',default=None)
    p.add_argument('--show',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    reaper=None
    if args.impl=='py' and np is None:
        args.impl='cpp'
    if args.impl=='cpp':
        reaper=find_reaper(args.reaper_bin)
    src=args.input_wav
    if args.self_test:
        if np is None:
            raise RuntimeError('Python 自测需要 numpy，请安装 numpy 或使用 --impl cpp')
        import wave, tempfile
        sr=16000
        dur=1.0
        freq=200.0
        t=np.arange(int(sr*dur))/sr
        x=(0.8*np.sin(2*math.pi*freq*t)).astype(np.float32)
        buf=(x*32767.0).astype(np.int16).tobytes()
        with tempfile.TemporaryDirectory() as td:
            test_wav=os.path.join(td,'test.wav')
            with wave.open(test_wav,'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(buf)
            process_file(args.impl,reaper,test_wav,args.frame_interval,args.min_f0,args.max_f0,args.hilbert,args.no_highpass,args.plot_out,args.show,args.csv_out,True)
        return
    if not src:
        raise SystemExit('需要提供 input_wav，或使用 --self-test')
    if os.path.isdir(src):
        out_dir=args.out_dir or os.path.join(src,'f0_csv')
        os.makedirs(out_dir,exist_ok=True)
        exts={'.wav','.WAV','.wave','.WAVE'}
        for root,dirs,files in os.walk(src):
            for name in files:
                base,ext=os.path.splitext(name)
                if ext in exts:
                    in_path=os.path.join(root,name)
                    rel=os.path.relpath(in_path,src)
                    safe=rel.replace(os.sep,'__')
                    plot_out=None if args.csv_only else os.path.join(out_dir,safe+'.png')
                    csv_out=os.path.join(out_dir,safe+'.csv')
                    try:
                        process_file(args.impl,reaper,in_path,args.frame_interval,args.min_f0,args.max_f0,args.hilbert,args.no_highpass,plot_out,args.show,csv_out,args.csv_only)
                    except Exception as e:
                        sys.stderr.write(str(e)+'\n')
        return
    process_file(args.impl,reaper,src,args.frame_interval,args.min_f0,args.max_f0,args.hilbert,args.no_highpass,args.plot_out,args.show,args.csv_out,args.csv_only)

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        sys.stderr.write(str(e)+'\n')
        sys.exit(1)

