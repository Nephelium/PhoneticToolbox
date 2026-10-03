"""In-memory EGG tables and Agg render adapter. Numerical rules live in core."""
import io
import numpy as np
import pandas as pd
from phonetic_core.egg import cq_segment
from phonetic_core.egg.metrics import calculate_cq_sq
from phonetic_core.egg.export_series import batch_columns, waveform_series, spectral_series


def f0_axis_range(values):
    """Display-only counterpart of visibleF0Range; no detector limit changes."""
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)&(values>0)]
    if not len(values):return None
    low,high=float(values.min()),float(values.max())
    pad=max(5,(high-low)*.1,high*.02)
    return max(0,float(np.floor(low-pad))),float(np.ceil(high+pad))


def cq_series(result, config, settings, start, end):
    values = (calculate_cq_sq(result.gci_times, result.goi_times, result.peak_times)
              if settings.mode == 'batch' else cq_segment(result, start, end, config, use_raw_signal=settings.signal_mode == 'raw'))
    return tuple(np.asarray(v if v is not None else [], dtype=float) for v in values)


def csv_bytes(result, config, settings, start, end):
    if settings.mode == 'batch':
        times, columns, mask = batch_columns(result, keep_praat=settings.keep_praat_f0,
            keep_gci=settings.keep_gci_f0, keep_reaper=settings.keep_reaper_f0, silence_threshold=settings.silence_threshold)
        frame = pd.DataFrame({'Time (s)': times, **columns})
        return frame.to_csv(index=False, na_rep='NaN', float_format='%.6f').encode(), int(mask.sum())
    times, cq, sq = cq_series(result, config, settings, start, end)
    take = (times >= start) & (times < end)
    frame = pd.DataFrame({'CQ': cq[take], 'SQ': sq[take]}, index=times[take])
    for keep, key, t, v in [(settings.keep_praat_f0, 'F0_Praat (Hz)', result.audio_f0_times, result.audio_f0_values),
                           (settings.keep_gci_f0, 'F0_GCI (Hz)', result.gci_f0_times, result.gci_f0_values)]:
        if keep or settings.mode == 'single':
            t = np.asarray(t if t is not None else []); v = np.asarray(v if v is not None else [])
            take = (t >= start) & (t < end)
            frame = frame.join(pd.DataFrame({key: v[take]}, index=t[take]), how='outer')
    if settings.glottal_movement:
        events = [(t, v) for t, v in result.glottal_movement_events if start <= t < end]
        frame = frame.join(pd.DataFrame({'Glottal_Movement': [v for _,v in events]}, index=[t for t,_ in events]), how='outer')
    if settings.keep_reaper_f0:
        t=np.asarray(result.reaper_f0_times);v=np.asarray(result.reaper_f0_values)
        take=(t>=start)&(t<end)
        frame=frame.join(pd.DataFrame({'F0_REAPER (Hz)':v[take]},index=t[take]),how='outer')
    return frame.sort_index().to_csv(na_rep='NaN', index_label='Time (s)').encode(), 0


def plot_bytes(result, config, settings, first, last, font_evidence=None):
    # Explicit fixed, print-friendly V2 exports. These are scientific artifacts,
    # not the interactive V3 page or its light/dark UI theme.
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    start, end = first/result.fs, last/result.fs
    batch = settings.mode == 'batch'
    size, dpi = ((12,6), 100) if batch else ((10,6), 150)
    style = {'font.family': ['Times New Roman', 'SimSun'], 'axes.unicode_minus': False,
             'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
             'axes.edgecolor': 'black', 'text.color': 'black', 'grid.color': '#DDDDDD', 'grid.linestyle': ':'}
    blobs = {}
    if settings.font is not None:
        from .fonts import resolve_fonts
        fonts=font_evidence or resolve_fonts(settings.font)
        style.update({'font.family':[fonts['latin']['family'],fonts['ipa']['family'],fonts['zh']['family']],
                      'font.size':settings.font.size_px*72/96,'axes.unicode_minus':True})
    def figure():
        fig = Figure(figsize=size); FigureCanvasAgg(fig); return fig
    def save(fig, name):
        # Explicit phonetic labels retain Doulos even if a Latin face contains
        # IPA glyphs. Current EGG axes are ordinary English/scientific labels.
        if settings.font is not None:
            import re
            from matplotlib.text import Text
            for text in fig.findobj(Text):
                if re.search(r'[\u0250-\u036f\u1d00-\u1dbf]|\[[a-zɑ-ʯ]+\]',text.get_text()):
                    text.set_fontfamily(['Doulos SIL',fonts['zh']['family']])
        stream = io.BytesIO(); fig.savefig(stream, format='png', dpi=dpi)
        blobs[name] = stream.getvalue(); fig.clear()
    with rc_context(style):
        fig = figure(); axis = fig.add_subplot(111); right = axis.twinx()
        times,cq,sq = cq_series(result,config,settings,start,end)
        if len(times):
            a, = axis.plot(times,cq,color='blue',marker='.',linestyle='',markersize=2 if batch else 5,label='CQ')
            b, = right.plot(times,sq,color='green',marker='x',linestyle='',markersize=2 if batch else 5,label='SQ')
            axis.legend([a,b],['CQ','SQ'],loc='upper right')
        axis.set(title='CQ / SQ',xlim=(start,end),ylim=(0,1),ylabel='CQ')
        right.set_ylabel('SQ')
        right.set_ylim(-1.1,1.1); axis.tick_params(axis='y',colors='blue'); right.tick_params(axis='y',colors='green')
        axis.grid(True); fig.tight_layout(); save(fig,'egg_CQ_SQ.png')

        fig = figure(); fig.set_layout_engine('constrained'); axis = fig.add_subplot(111); right = axis.twinx()
        power, frequencies, bins = spectral_series(result.audio_signal[first:last], result.fs, config.spec_window_ms)
        # The fixed export axis shows 0–5000 Hz. Crop invisible image rows before
        # RGBA allocation, preserving the original imshow pixel boundaries.
        cell_height=(frequencies[-1]-frequencies[0])/len(frequencies)
        rows=min(len(frequencies),int(np.ceil(5000/cell_height))+2)
        visible=power[:rows].copy();del power
        with np.errstate(divide='ignore'): np.log10(visible,out=visible)
        visible*=10;db=visible
        # V2 single-file extent stretches the selected PSD to the ROI boundaries.
        # Retain that presentation while waveforms and exported CSV use sample time.
        extent = [start,end,frequencies[0],frequencies[0]+rows*cell_height]
        if batch:
            nfft = int(result.fs*config.spec_window_ms/1000)
            pad = (nfft-int(nfft*.75))/result.fs/2
            extent[:2] = [bins[0]-pad,bins[-1]+pad]
        image = axis.imshow(np.flipud(db), origin='upper', aspect='auto', extent=extent,
                            cmap='gray_r',vmin=config.spec_vmin,vmax=config.spec_vmax)
        fig.colorbar(image,ax=axis,label='Magnitude (dB)')
        visible_f0=[]
        for keep,t,v,color in [(settings.keep_praat_f0,result.audio_f0_times,result.audio_f0_values,'black'),
                               (settings.keep_gci_f0,result.gci_f0_times,result.gci_f0_values,'red'),
                               (settings.keep_reaper_f0,result.reaper_f0_times,result.reaper_f0_values,'#007a70')]:
            if keep and t is not None:
                take = (t >= start) & (t < end) & np.isfinite(v) & (v>0)
                visible_f0.extend(v[take]);right.plot(t[take],v[take],color=color,marker='.',markersize=2,linestyle='None')
        axis.set(title='Spectrogram / F0',xlim=(start,end),ylim=(0,5000),ylabel='Hz')
        limits=f0_axis_range(visible_f0)
        if limits:right.set_ylim(*limits);right.set_ylabel('F0 (Hz)')
        else:right.yaxis.set_visible(False)
        save(fig,'egg_SPEC_F0.png')

        fig = figure(); top = fig.add_subplot(211); bottom = fig.add_subplot(212,sharex=top)
        times,audio,values = waveform_series(result,config,first,last,raw=settings.signal_mode=='raw',batch=batch)
        top.plot(times,audio,color='black',lw=.5); bottom.plot(times,values,color='black',lw=.5)
        top.set_title('Audio',loc='center');bottom.set_title('EGG',loc='center')
        top.set_ylabel('Amplitude');bottom.set_ylabel('Amplitude');bottom.set_xlabel('Time (s)')
        top.grid(True); bottom.grid(True); bottom.set_xlim(start,end); fig.tight_layout(); save(fig,'egg_WAVEFORMS.png')
    return blobs
