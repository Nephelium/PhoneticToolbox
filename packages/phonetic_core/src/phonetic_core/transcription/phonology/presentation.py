"""Explicit V3 presentation fixes, separate from the exact V2 migration."""
from io import BytesIO
from copy import copy
from openpyxl import load_workbook
from openpyxl.cell.rich_text import CellRichText
from .render import PhonologyRenderer


class WorkbenchRenderer(PhonologyRenderer):
    def __init__(self,font):
        super().__init__()
        if font.get('ipa')!='Doulos SIL':raise ValueError('m14_ipa_font')
        self.font=font;self.ordered_tones=[];self.tone_map={}

    def _write_matrix_xlsx(self,analysis,tone_class_map,output_path,ordered_initials,ordered_finals,ordered_tones):
        self.ordered_tones=ordered_tones;self.tone_map=tone_class_map
        temporary=BytesIO()
        super()._write_matrix_xlsx(analysis,tone_class_map,temporary,ordered_initials,ordered_finals,ordered_tones)
        w=load_workbook(BytesIO(temporary.getvalue()),rich_text=True);s=w.active
        for row in s:
            for c in row:
                f=copy(c.font);f.name=self.font['ipa'] if (c.row==1 or c.column==1) and c.coordinate!='A1' else self.font['zh'];f.sz=self.font['size_px']*.75;c.font=f
        # FIX04: V2 fixed 52 pt height clips long homophone cells. Allow wrapping.
        for row in s:
            length=max((sum(2 if self._contains_cjk(ch) else 1 for ch in str(c.value or '')) for c in row),default=0)
            s.row_dimensions[row[0].row].height=min(409,max(36 if row[0].row==1 else 52,((length+23)//24)*self.font['size_px']*1.1))
        s.print_options.horizontalCentered=True
        s.sheet_properties.pageSetUpPr.fitToPage=True
        s.page_setup.orientation='landscape';s.page_setup.paperSize=s.PAPERSIZE_A4;s.page_setup.fitToWidth=1;s.page_setup.fitToHeight=0
        saved=BytesIO();w.save(saved)
        # FIX05: openpyxl 3.1.5 omits xml:space on a whitespace-only rich run.
        # Excel 16 refuses these V2 workbooks (openpyxl itself can read them).
        # Preserve our explicit inter-tone separator without changing cell text.
        from zipfile import ZipFile,ZIP_DEFLATED
        with ZipFile(BytesIO(saved.getvalue())) as source,ZipFile(output_path,'w',ZIP_DEFLATED) as target:
            for name in source.namelist():
                data=source.read(name)
                if name.startswith('xl/worksheets/'):
                    data=data.replace(b'<t> </t>',b'<t xml:space="preserve"> </t>')
                target.writestr(name,data)

    def _build_rich_text_cell_value(self,tone_map):
        # FIX01: preserve the same configured tone order as both Word documents.
        rich=CellRichText()
        for i,label in enumerate(self._ordered_tone_labels(set(tone_map),self.tone_map,self.ordered_tones)):
            if i:self._append_xlsx_rich_text(rich,' ',self.font['latin'],11)
            self._append_xlsx_rich_text(rich,f'[{label}]',self.font['zh'],11)
            for row in tone_map[label]:
                self._append_xlsx_mixed_text(rich,row.character,11)
                if row.note:self._append_xlsx_mixed_text(rich,row.note,11,True)
        return rich

    def _append_xlsx_rich_text(self,rich_text,text,font_name,size,subscript=False):
        if font_name=='宋体':font_name=self.font['zh']
        elif font_name=='Times New Roman':font_name=self.font['latin']
        super()._append_xlsx_rich_text(rich_text,text,font_name,self.font['size_px']*.75,subscript)

    def _setup_word_document_styles(self,doc):
        super()._setup_word_document_styles(doc)
        from docx.shared import Pt
        from docx.oxml.ns import qn
        style=doc.styles['Normal'];style.font.name=self.font['latin'];style.font.size=Pt(self.font['size_px']*.75)
        style._element.rPr.rFonts.set(qn('w:eastAsia'),self.font['zh'])

    def _apply_word_run_style(self,run,size=12,bold=False,subscript=False,cjk=False):
        super()._apply_word_run_style(run,size,bold,subscript,cjk)
        from docx.shared import Pt
        from docx.oxml.ns import qn
        # Original non-CJK runs carry sound symbols and their statistics labels.
        run.font.name=self.font['zh'] if cjk else self.font['ipa']
        run.font.size=Pt(self.font['size_px']*.75)
        run._element.rPr.rFonts.set(qn('w:eastAsia'),self.font['zh'])
