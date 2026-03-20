import os
import glob
import re
import numpy as np
from itertools import product
from parselmouth.praat import call

def generate_batch_linear(snd, snd_path, times, original_f0, xmin, xmax,
                          t1, t2, f1_list, f2_list, 
                          knot_points, start_mode, end_mode, knot_modes, 
                          offset_mode=False):
    """
    Generate batch F0 modifications based on linear interpolation and knot points.
    Saves the generated files to disk.
    
    Args:
        snd: Parselmouth Sound object.
        snd_path: Path to the original sound file.
        times: Array of time points.
        original_f0: Array of original F0 values.
        xmin, xmax: Current view range (to define the segment to modify).
        t1, t2: Start and end times for the modification ramp.
        f1_list, f2_list: Start and end frequency lists.
        knot_points: List of dicts with 'time' and 'freqs'.
        start_mode, end_mode: Connection modes for start and end points.
        knot_modes: List of connection modes for knot points.
        offset_mode: Boolean, whether to use offset mode.
        
    Returns:
        str: Success message.
    """
    if snd is None:
        raise ValueError("音频未加载")
        
    if t2 <= t1:
        raise ValueError("终止时间必须大于起始时间")
    if not f1_list or not f2_list:
        raise ValueError("起止基频列表不能为空")

    folder = os.path.dirname(snd_path)
    stem, _ = os.path.splitext(os.path.basename(snd_path))
    base_pattern = f"{stem}_{xmin:.2f}_{xmax:.2f}_modified"
    
    # Check existing files to determine index
    existing = glob.glob(os.path.join(folder, f"{base_pattern}_*.wav"))
    def get_index(fname):
        m = re.search(r"_(\d+)\.wav$", fname)
        return int(m.group(1)) if m else 0
    cur_max = max([get_index(f) for f in existing], default=0)
    idx_counter = cur_max

    # Prepare knot data
    knot_times = [kp["time"] for kp in knot_points]
    knot_freq_lists = [kp["freqs"] for kp in knot_points]
    
    all_times = [t1] + knot_times + [t2]
    
    # Validate times
    for kt in knot_times:
        if kt <= t1 or kt >= t2:
            raise ValueError("拐点时间需在起止时间之间")
    if len(set(all_times)) != len(all_times):
        raise ValueError("时间点不能重复")

    # Combine all points and modes
    point_lists = [f1_list] + knot_freq_lists + [f2_list]
    
    current_knot_modes = []
    for i in range(len(knot_freq_lists)):
        if i < len(knot_modes):
            current_knot_modes.append(knot_modes[i])
        else:
            current_knot_modes.append("order")
            
    point_modes = [start_mode] + current_knot_modes + [end_mode]

    # Pre-process lists based on mode (handle 'constant')
    processed_lists = []
    for lst, mode in zip(point_lists, point_modes):
        if mode == "constant":
            if not lst:
                raise ValueError("常量模式频率列表不能为空")
            processed_lists.append([lst[0]])
        else:
            processed_lists.append(lst)

    # Validate diagonal lengths
    diag_indices = [i for i, m in enumerate(point_modes) if m in ("order", "reverse")]
    if diag_indices:
        base_len = len(processed_lists[diag_indices[0]])
        for di in diag_indices[1:]:
            if len(processed_lists[di]) != base_len:
                raise ValueError("对角模式(顺序/逆序)的列表长度必须一致")
    else:
        base_len = 1

    # Handle full connection
    full_indices = [i for i, m in enumerate(point_modes) if m == "full" and len(processed_lists[i]) > 1]
    full_product_source = [processed_lists[i] for i in full_indices]
    full_product = list(product(*full_product_source)) if full_indices else [()]

    # Generate combinations
    combos = []
    for k in range(base_len):
        for fp in full_product:
            fp_iter = iter(fp)
            path = []
            for idx in range(len(processed_lists)):
                mode = point_modes[idx]
                if mode in ("order", "reverse"):
                    dlst = processed_lists[idx]
                    sorted_vals = sorted(dlst)
                    if mode == "reverse":
                        sorted_vals = list(reversed(sorted_vals))
                    path.append(sorted_vals[k])
                elif mode == "full":
                    if len(processed_lists[idx]) == 1:
                        path.append(processed_lists[idx][0])
                    else:
                        path.append(next(fp_iter))
                else:  # constant
                    path.append(processed_lists[idx][0])
            combos.append(tuple(path))

    # Prepare data for synthesis
    mask_win = (times >= xmin) & (times <= xmax)
    win_times = times[mask_win]
    mask_seg = (times >= t1) & (times <= t2)
    seg_times = times[mask_seg]
    
    if seg_times.size == 0:
        raise ValueError("时间范围无数据")

    part_snd = snd.extract_part(from_time=xmin, to_time=xmax, preserve_times=True)

    # Generate files
    for i, vals in enumerate(combos, start=1):
        ctrl_vals = list(vals)
        new_pitch_tier = call("Create PitchTier", "modified", xmin, xmax)
        delta_curve = np.zeros_like(seg_times)
        
        # Linear interpolation between control points
        for si in range(len(all_times) - 1):
            ta = all_times[si]
            tb = all_times[si + 1]
            va = ctrl_vals[si]
            vb = ctrl_vals[si + 1]
            seg_mask = (seg_times >= ta) & (seg_times <= tb)
            if tb - ta > 0:
                delta_curve[seg_mask] = va + (vb - va) * (seg_times[seg_mask] - ta) / (tb - ta)
        
        base_curve_win = original_f0[mask_win].copy()
        seg_in_win = (win_times >= t1) & (win_times <= t2)
        
        if offset_mode:
            seg_final = original_f0[mask_seg] + delta_curve
        else:
            seg_final = delta_curve
            
        base_curve_win[seg_in_win] = seg_final
        
        # Add points to PitchTier
        for t, f in zip(win_times, base_curve_win):
            call(new_pitch_tier, "Add point", t, f)
            
        # Ensure control points are added explicitly
        for si, ta in enumerate(all_times):
            # Find closest original F0 if offset mode
            base_val = original_f0[(np.abs(times - ta)).argmin()]
            final_val = ctrl_vals[si] if not offset_mode else (base_val + ctrl_vals[si])
            call(new_pitch_tier, "Add point", ta, final_val)
            
        manipulation = call(part_snd, "To Manipulation", 0.01, 75, 600)
        call([manipulation, new_pitch_tier], "Replace pitch tier")
        out_snd = call(manipulation, "Get resynthesis (overlap-add)")
        
        idx_counter += 1
        
        # Construct filename
        mode_tag = "lin"
        off_tag = "_offset" if offset_mode else ""
        knot_detail = ""
        if knot_times:
            pairs = [f"k{kt:.2f}_{kv:.1f}Hz" for kt, kv in zip(knot_times, ctrl_vals[1:-1])]
            knot_detail = "_" + "_".join(pairs)
        freq_path = "-".join([f"{v:.1f}" for v in ctrl_vals])
        tag = f"seg_{t1:.2f}-{t2:.2f}_kn{len(knot_times)}_{mode_tag}{off_tag}_Fpath_{freq_path}{knot_detail}_combo{i}"
        save_name = f"{base_pattern}_{tag}_{idx_counter}.wav"
        
        out_snd.save(os.path.join(folder, save_name), "WAV")
        
    return "批量生成完成"
