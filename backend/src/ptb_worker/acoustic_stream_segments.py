"""Slice committed same-source bounded results, preserving both native grids."""
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from phonetic_core.segmentation import plan_segments
from .io.annotations import decode_textgrid
from .io.parameter_bundle import open_checked,BundleWriter,export_xlsx,quote,digest
from .acoustic_stream_child import atomic


def run_segments(root,request):
    connection,meta=open_checked(root/'parent_table.bin')
    try:
        with sf.SoundFile(root/'input.wav') as source:
            if meta['audio_sha256']!=request['audio_sha256'] or meta['sample_rate_hz']!=source.samplerate or meta['channels']!=source.channels:
                raise ValueError('parent_result_source_mismatch')
            plan=plan_segments(decode_textgrid((root/'textgrid.bin').read_bytes()),request['layer'],source.samplerate,source.frames)
            if not plan:raise ValueError('no_labelled_segments')
            def clean(text):return ''.join(c for c in text if c.isalnum() or c in '._-').strip('.')[:36] or 'label'
            names=[];segments=[]
            for index,part in enumerate(plan):
                first,last=part.first/source.samplerate,part.last/source.samplerate
                base=f'{clean(Path(request["audio_name"]).stem)}_{clean(request["layer"])}_{clean(part.label)}_{part.start_s:.3f}_{part.end_s:.3f}_{part.index+1:04d}'
                source.seek(part.first)
                with sf.SoundFile(root/(base+'.wav'),'w',samplerate=source.samplerate,channels=source.channels,subtype=source.subtype) as audio:
                    remaining=part.last-part.first
                    while remaining:
                        n=min(65536,remaining);audio.write(source.read(n,dtype='float64',always_2d=True));remaining-=n
                writer=BundleWriter(root/(base+'.ptb.sqlite'))
                try:
                    for table,info in meta['tables'].items():
                        if 'Source_Time_s' in info['columns']:raise ValueError('legacy_parameter_time_mismatch')
                        columns=info['columns'];types=dict(zip(columns,info['kinds']))
                        query='SELECT * FROM '+table+' WHERE Time_s>=? AND Time_s<? ORDER BY Time_s'
                        cursor=connection.execute(query,(first,last));empty=True
                        while True:
                            rows=cursor.fetchmany(2000)
                            if not rows:break
                            empty=False;arrays={c:np.array([r[i] for r in rows],dtype=object if types[c]=='text' else float) for i,c in enumerate(columns)}
                            arrays['Source_Time_s']=arrays['Time_s'].copy()
                            for c in ('Time_s','GCI_s','GOI_s','Next_GCI_s','F0_Time_s'):
                                if c in arrays:arrays[c]=arrays[c]-first
                            writer.append(table,arrays)
                        if empty:writer.append(table,{c:np.array([],dtype=object if types.get(c)=='text' else float) for c in [*columns,'Source_Time_s']})
                    writer.finish({**meta,'duration_s':last-first,'source_offset_s':first,'parent_audio_sha256':meta['audio_sha256'],
                        'audio_sha256':digest(root/(base+'.wav')),'reestimated':False,'time_policy':'source_frames_and_cycles_rebased_to_first_sample'})
                finally:writer.conn.close()
                export_xlsx(root/(base+'.ptb.sqlite'),root/(base+'.xlsx'))
                names.extend(base+suffix for suffix in ('.wav','.xlsx','.ptb.sqlite'))
                segments.append(dict(label=part.label,first_sample=part.first,last_sample=part.last))
                atomic(root/'status.json',dict(progress=.1+.83*(index+1)/len(plan)))
            atomic(root/'segments.ptb.json',dict(format_revision='m01-bundle/2',audio_sha256=request['audio_sha256'],segments=segments,reestimated=False))
            return dict(success=True,files=[*names,'segments.ptb.json'])
    finally:connection.close()
