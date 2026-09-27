"""One-time mechanical source split. Refuses to overwrite files; no V2 mutation."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT.parent/'PhoneticToolbox_v2'
target = ROOT/'packages/phonetic_core/src/phonetic_core/transcription/phonology'
target.mkdir(parents=True,exist_ok=True)
notice = '# Direct V2 migration; source_id=PENDING-PHONOLOGY. See NOTICE.md and M14-source-map.md.\n'
def write(name,text):
    with (target/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)

parser=(V2/'phonetic_toolbox/core/transcription/phonology_induction.py').read_text(encoding='utf-8')
klatt=ast.parse((V2/'phonetic_toolbox/core/synthesis/klatt/input_parser.py').read_text(encoding='utf-8'))
vowels=next(ast.literal_eval(n.value) for n in klatt.body if isinstance(n,ast.Assign) and n.targets[0].id=='VOWEL_FORMANTS')
parser=parser.replace('from phonetic_toolbox.core.synthesis.klatt.input_parser import VOWEL_FORMANTS','VOWEL_SYMBOLS = '+repr(tuple(vowels)))
parser=parser.replace('set(VOWEL_FORMANTS.keys())','set(VOWEL_SYMBOLS)')
write('parser.py',notice+parser)
write('models.py',notice+(V2/'phonetic_toolbox/models/phonology_models.py').read_text(encoding='utf-8'))
source=(V2/'phonetic_toolbox/services/phonology_service.py').read_text(encoding='utf-8'); lines=source.splitlines()
cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='PhonologyInductionService')
io={'load_rows','export_outputs','_load_rows_from_excel','_load_rows_from_csv','_load_rows_from_text','_rows_from_frame'}
export={'_write_word_document','_resolve_document_factory','_append_summary_sections','_pick_examples','_append_word_entries','_write_matrix_xlsx','_ordered_tone_labels','_append_two_column_entries','_build_rich_text_cell_value','_append_xlsx_mixed_text','_append_xlsx_rich_text','_setup_word_document_styles','_center_word_paragraph','_style_word_paragraph_runs','_add_word_text','_apply_word_run_style','_contains_cjk','_render_plain_entry','_to_subscript_text'}
rules=[]; renders=[]
for n in cls.body:
    name=getattr(n,'name','')
    if name in io: continue
    if name=='_resolve_document_factory':
        renders.append('    def _resolve_document_factory(self):\n        from docx import Document\n        return Document');continue
    text='\n'.join(lines[n.lineno-1:n.end_lineno])
    (renders if name in export else rules).append(text)
write('rules.py',notice+'from __future__ import annotations\nimport re\nfrom collections import defaultdict\nfrom .parser import PhonologyInductionParser\nfrom .models import PhonologyInputRow, ParsedPhonologyRow, PhonologyAnalysisResult\n\nclass PhonologyRules:\n'+'\n\n'.join(rules)+'\n')
write('render.py',notice+'from __future__ import annotations\nfrom io import BytesIO\nfrom openpyxl import Workbook\nfrom openpyxl.styles import Alignment, Border, Font, PatternFill, Side\nfrom openpyxl.utils import get_column_letter\nfrom openpyxl.cell.rich_text import CellRichText, TextBlock\nfrom openpyxl.cell.text import InlineFont\nfrom .models import ParsedPhonologyRow, PhonologyAnalysisResult\nfrom .rules import PhonologyRules\n_OPENPYXL_RICH_TEXT_AVAILABLE=True\n\nclass PhonologyRenderer(PhonologyRules):\n'+'\n\n'.join(renders).replace('output_path: Path','output_path: BytesIO')+'\n')
write('__init__.py',notice+'from .rules import PhonologyRules\nfrom .models import PhonologyInputRow, ParsedPhonologyRow, PhonologyAnalysisResult\n')
write('NOTICE.md','M14 is directly migrated from PhoneticToolbox v2 parser, models and service.\nSource ID: PENDING-PHONOLOGY. No new authorship or license claim.\nThe earlier code and method provenance remains unresolved; see docs/modules/evidence/M14-source-map.md.\nThe exact vowel keys were copied from the V2 Klatt input parser; no synthesis dependency remains.\n')
print(target)
