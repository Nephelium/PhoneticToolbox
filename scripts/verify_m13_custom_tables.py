"""Owned synthetic two-column files and production browser acceptance."""
import json
import os
from pathlib import Path
import subprocess
import zipfile
from verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[1]


def fixtures(scratch):
    (scratch / 'hanzi.csv').write_text('汉字,国际音标\n行,hɑŋ˧˥\n行,ɕiŋ˧˥\n𠮷,kiʔ\n', encoding='utf-8-sig')
    (scratch / 'invalid.csv').write_text('汉字,国际音标\n银行,a\n', encoding='utf-8')
    (scratch / 'replacement.csv').write_text('汉字,国际音标\n行,a˥\n行,b˩\n', encoding='utf-8')
    (scratch / 'syllables.tsv').write_text('拼音\tIPA\nma\tmɑ\nmā\tma˥\nma5\tmə\nnv3\tny˨˩˦\n', encoding='utf-8')
    (scratch / 'wide.tsv').write_text('调查字表\n\n编号\t音标\t备注\t汉字\n1\tma55\t备注甲\t妈\n2\tma³⁵\t备注乙\t麻\n3\tma２１４\t备注丙\t马\n', encoding='utf-8')
    (scratch / 'classes.tsv').write_text('汉字\tIPA\n妈\tma¹\n麻\tma2\n马\tma３\n', encoding='utf-8')
    main = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    sheets = [
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row><row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2" t="s"><v>3</v></c></row>',
        '<row r="1"><c r="A1" t="inlineStr"><is><t>汉字</t></is></c><c r="B1" t="inlineStr"><is><t>国际音标</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>女</t></is></c><c r="B2" t="inlineStr"><is><t>nɯ˩</t></is></c></row>']
    for name, compression in [('shared.xlsx', zipfile.ZIP_DEFLATED), ('stored.xlsx', zipfile.ZIP_STORED)]:
        with zipfile.ZipFile(scratch / name, 'w', compression=compression) as archive:
            archive.writestr('[Content_Types].xml', '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>')
            archive.writestr('_rels/.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
            archive.writestr('xl/workbook.xml', f'<workbook xmlns="{main}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="共享字符串" sheetId="1" r:id="rId1"/><sheet name="调查字表" sheetId="2" r:id="rId2"/></sheets></workbook>')
            archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="/xl/worksheets/sheet2.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/></Relationships>')
            archive.writestr('xl/sharedStrings.xml', f'<sst xmlns="{main}"><si><t>汉字</t></si><si><t>国际音标</t></si><si><t>妈</t></si><si><t>mɑ˥</t></si></sst>')
            for i, data in enumerate(sheets, 1):
                archive.writestr(f'xl/worksheets/sheet{i}.xml', f'<worksheet xmlns="{main}"><sheetData>{data}</sheetData></worksheet>')
    return {'files': ['hanzi.csv', 'invalid.csv', 'replacement.csv', 'syllables.tsv', 'shared.xlsx', 'stored.xlsx', 'wide.tsv', 'classes.tsv'], 'source': 'synthetic Chinese/IPA, arbitrary columns and leading rows, numeric/superscript/fullwidth tones; shared/inline strings, stored/deflated ZIP, two sheets'}


if __name__ == '__main__':
    with verification_run('m13-custom-tables', 'chrome', 'Production AppShell; synthetic CSV/TSV/XLSX; isolated persistent Chrome, restart, polyphones, PNG and help') as (out, scratch):
        (out / 'inputs.json').write_text(json.dumps(fixtures(scratch), ensure_ascii=False, indent=2), encoding='utf-8')
        result = subprocess.run(['node', str(ROOT / 'tests/e2e/m13-custom-tables.cjs'), str(out), str(scratch)], cwd=ROOT,
                                capture_output=True, text=True, encoding='utf-8', errors='replace', creationflags=subprocess.CREATE_NO_WINDOW)
        (out / 'browser.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        print(out, flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
            raise RuntimeError('M13 custom table browser acceptance failed')
