"""Pure in-memory three-file export. Platform owns paths and atomic publication."""
from io import BytesIO
from .render import PhonologyRenderer

NAMES=('同音字表_韵母到声母.docx','同音字表_声母到韵母.docx','同音字表_二维表.xlsx')


def export(analysis,tone_class_map,tone_order,initial_order,final_order,*,renderer=None,cancelled=lambda:False):
    r=renderer or PhonologyRenderer()
    if not analysis.rows:raise ValueError('m14_no_valid_rows')
    if len(analysis.rows)>10_000 or len(analysis.unique_initials)*len(analysis.unique_finals)>20_000:raise ValueError('m14_output_shape_budget')
    initials=r._resolve_order(analysis.unique_initials,initial_order)
    finals=r._resolve_order(analysis.unique_finals,final_order)
    tones=r._resolve_order(analysis.unique_tones,tone_order)
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
