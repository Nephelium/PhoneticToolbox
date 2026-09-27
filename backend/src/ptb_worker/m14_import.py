"""M14 authorized bytes -> V2-compatible rows; platform decoding only."""
import io
from pathlib import PurePath
import re
from zipfile import ZipFile, BadZipFile
import pandas as pd
from phonetic_core.transcription.phonology import PhonologyRules

MAX_INPUT = 2_000_000
MAX_ROWS = 10_000
MAX_CELL = 512
EXTENSIONS = ('.xlsx', '.xls', '.csv', '.txt', '.tsv')


def load(raw: bytes, name: str, skip_first_row: bool = True):
    if not isinstance(raw,bytes) or not raw: raise ValueError('m14_empty_input')
    if len(raw)>MAX_INPUT: raise ValueError('m14_input_budget')
    suffix=PurePath(name).suffix.lower()
    if suffix not in EXTENSIONS: raise ValueError('m14_unsupported_format')
    rules=PhonologyRules(); rows=[]; skipped=[]; total=0
    def accept(cols,index):
        nonlocal total
        total+=1
        if total>MAX_ROWS+1: raise ValueError('m14_row_budget')
        if any(len(v)>MAX_CELL or any(ord(c)<32 and c not in '\t\r\n' for c in v) for v in cols): raise ValueError('m14_invalid_cell')
        if skip_first_row and index==0: skipped.append(dict(row=index+1,reason='首行'));return
        row=rules._parse_columns(cols)
        if row:rows.append(row)
        else:skipped.append(dict(row=index+1,reason='表头、空行或缺少字/音标'))
    try:
        if suffix=='.xlsx':
            with ZipFile(io.BytesIO(raw)) as archive:
                if len(archive.infolist())>512 or sum(i.file_size for i in archive.infolist())>32_000_000:raise ValueError('m14_expanded_budget')
                # Formula cache values are not reproducible input. Reject explicitly.
                for entry in archive.infolist():
                    if entry.filename.startswith('xl/worksheets/') and entry.filename.endswith('.xml'):
                        if re.search(rb'<(?:\w+:)?f(?:\s|>)',archive.read(entry)):raise ValueError('m14_formula_input')
        if suffix in ('.xlsx','.xls','.csv'):
            stream=io.BytesIO(raw)
            if suffix=='.csv': frame=pd.read_csv(stream,header=None,dtype=str,encoding='utf-8-sig',nrows=MAX_ROWS+2)
            else:frame=pd.read_excel(stream,header=None,dtype=str,nrows=MAX_ROWS+2,engine='xlrd' if suffix=='.xls' else 'openpyxl')
            if len(frame.columns)<2:raise ValueError('m14_missing_columns')
            if len(frame.columns)>64:raise ValueError('m14_column_budget')
            for index,values in frame.iterrows():accept(['' if pd.isna(v) else str(v) for v in values.tolist()],index)
        else:
            for index,line in enumerate(raw.decode('utf-8-sig').splitlines()):
                accept(re.split(r'[\t,，]',line.strip(),maxsplit=2),index)
    except (ValueError,UnicodeError,BadZipFile, OSError, ImportError) as e:
        if str(e).startswith('m14_'):raise
        raise ValueError('m14_decode_failed') from e
    if not rows:raise ValueError('m14_no_valid_rows')
    duplicates=len(rows)-len(set((r.character,r.ipa,r.note) for r in rows))
    return rows,dict(total_rows=total,accepted_rows=len(rows),skipped=skipped,duplicate_rows=duplicates)
