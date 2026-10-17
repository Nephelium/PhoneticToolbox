"""Explicit public smoke check with task-owned synthetic identities and files."""
import argparse
import subprocess
from pathlib import Path
from verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--origin', required=True)
    p.add_argument('--account-file', type=Path, required=True)
    args = p.parse_args()
    if args.origin != 'https://www.phonetictoolbox.com':
        p.error('This check targets only the explicitly authorized public website')
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
    with verification_run('web-public', 'browser', 'Owned synthetic account; actual public HTTPS, Secure Cookie, settings, Linux LPC, paper PDF and logout') as (out, scratch):
        writer = PdfWriter()
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 18 Tf 50 700 Td (Public phonetics smoke fixture) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        writer.write(scratch/'fixture.pdf')
        writer.close()
        result = subprocess.run(['node', str(ROOT/'tests/e2e/web-public.cjs'), args.origin,
                                 str(args.account_file.resolve()), str(out), str(scratch)], cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)
