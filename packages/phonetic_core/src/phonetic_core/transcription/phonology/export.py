"""Pure in-memory three-file export. Platform owns paths and atomic publication."""
from io import BytesIO
from .render import PhonologyRenderer
from .models import NAMES


def export(analysis,tone_class_map,tone_order,initial_order,final_order,*,renderer=None,cancelled=lambda:False):
    r=renderer or PhonologyRenderer()
    if not analysis.rows:raise ValueError('m14_no_valid_rows')
    if len(analysis.rows)>10_000 or len(analysis.unique_initials)*len(analysis.unique_finals)>20_000:raise ValueError('m14_output_shape_budget')
    initials=r._resolve_order(analysis.unique_initials,initial_order)
    finals=r._resolve_order(analysis.unique_finals,final_order)
    tones=r._resolve_order(analysis.unique_tones,tone_order)
    # Excel's cell limit is a file-format constraint; reject before any output
    # is published rather than silently truncating homophone entries.
    cells={}
    labels={}
    for row in analysis.rows:
        key=(row.initial,row.final)
        cells[key]=cells.get(key,0)+len(row.character)+len(row.note)
        labels.setdefault(key,set()).add(tone_class_map.get(row.tone_value,row.tone_value))
    if any(size+sum(len(t)+3 for t in labels[key])>32767 for key,size in cells.items()):
        raise ValueError('m14_cell_output_budget')
    payloads={}
    for name,mode in zip(NAMES,('final_initial','initial_final',None)):
        if cancelled():raise InterruptedError('m14_cancelled')
        stream=BytesIO()
        kwargs=dict(ordered_initials=initials,ordered_finals=finals,ordered_tones=tones)
        if mode:r._write_word_document(analysis,tone_class_map,stream,mode=mode,**kwargs)
        else:r._write_matrix_xlsx(analysis,tone_class_map,stream,**kwargs)
        payloads[name]=stream.getvalue()
        if sum(map(len,payloads.values()))>16_000_000:raise ValueError('m14_output_budget')
    if cancelled():raise InterruptedError('m14_cancelled')
    return payloads
