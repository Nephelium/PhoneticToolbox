"""Explicit roots reserved for disposable Windows final-byte QA."""
from pathlib import Path

DEFAULT_QA_ROOT = Path('D:/PTB-Compact-QA-20261006')
QA_ROOTS = (DEFAULT_QA_ROOT, Path('F:/PTB-Compact-QA-20261024'))


def qa_root(value):
    root = Path(value).absolute()
    if root not in QA_ROOTS:
        raise ValueError('Expected an explicitly reserved D or F drive QA root')
    for part in (root, *root.parents):
        if part.exists() and (part.is_symlink() or getattr(part.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Linked QA root: ' + str(part))
    return root


def qa_output(value, root):
    base = qa_root(root)
    output = Path(value).absolute()
    if '..' in output.parts or output == base or not output.is_relative_to(base):
        raise ValueError('Expected an owned child of the selected QA root')
    for part in (output, *output.parents):
        if part.exists() and (part.is_symlink() or getattr(part.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Linked QA output: ' + str(part))
    return output
