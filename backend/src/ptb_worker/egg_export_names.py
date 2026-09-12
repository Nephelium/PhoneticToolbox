"""Portable save suggestions, independent of internal managed asset names."""
import hashlib
import re


def export_names(input_name, mode, start, end, names):
    original=str(input_name)
    stem=re.sub(r'\.[^.]*$', '', original)
    stem=re.sub(r'[\x00-\x1f/\\:<>"|?*]', '_', stem).strip(' .') or 'EGG'
    if len(stem)>150:
        stem=stem[:140]+'-'+hashlib.sha256(original.encode()).hexdigest()[:8]
    # A prefix also protects Windows device names when used without a suffix.
    if stem.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}:
        stem='EGG-'+stem
    if mode!='batch':
        stamp=lambda value:f'{value:.2f}s'.replace('.','_')
        stem+=f'_{stamp(start)}_{stamp(end)}'
    return {name:stem+name[len('egg'):] for name in names}
