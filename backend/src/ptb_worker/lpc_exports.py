"""Print-friendly LPC rendering and safe suggested export names."""
import io
import re
from pathlib import PurePath


def export_names(input_name, label, start, end):
    from ptb_api.lpc_models import LPC_NAMES
    stem = PurePath(input_name).stem
    # Keep IPA and Chinese; result assets themselves use fixed internal names.
    text = re.sub(r'[\x00-\x1f/\\:<>"|?*]', '_', stem+'_'+label).strip(' ._')[:100] or 'LPC'
    text = 'LPC_'+text+f'_{start:.6f}-{end:.6f}s'
    return {n:text+n[3:] for n in LPC_NAMES}


def plot_bytes(result, settings, label, fonts):
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    style = {'font.family':[fonts['latin']['family'],fonts['ipa']['family'],fonts['zh']['family']],
        'font.size':settings.font.size_px*72/96,'axes.unicode_minus':True,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
        'text.color':'black','axes.labelcolor':'black','axes.edgecolor':'black',
        'xtick.color':'black','ytick.color':'black','axes.titlesize':settings.font.size_px*72/96}
    with rc_context(style):
        fig=Figure(figsize=(8,4.5),dpi=300);FigureCanvasAgg(fig)
        try:
            axis=fig.add_subplot(111)
            axis.plot(result.frequencies_hz,result.magnitude_db,color='black',linewidth=1)
            axis.set(xlim=(0,settings.freq_max_hz),ylim=(result.amp_min_db,result.amp_max_db),
                     xlabel='Frequency (Hz)',ylabel='Amplitude (dB)')
            # Annotation is a phonetic label, including plain ASCII IPA.
            if label:axis.set_title(label,fontfamily=['Doulos SIL',fonts['zh']['family']])
            axis.grid(True,color='#dddddd',linestyle=':')
            fig.tight_layout()
            stream=io.BytesIO();fig.savefig(stream,format='png',dpi=300)
            return stream.getvalue()
        finally:fig.clear()
