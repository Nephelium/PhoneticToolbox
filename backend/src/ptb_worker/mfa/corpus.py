"""M11 corpus admission; pairing does not perform ASR or change transcription."""
from pathlib import Path
from .components import safe_name


def validate_corpus(root):
    root = Path(root).resolve()
    entries = list(root.rglob('*'))
    if any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)() for p in entries):
        raise ValueError('m11_corpus_link')
    audio = sorted(p for p in entries if p.is_file() and p.suffix.lower() == '.wav')
    if not audio:
        raise ValueError('m11_empty_corpus')
    if len(audio) > 100:
        raise ValueError('m11_corpus_budget')
    pairs = []
    for wav in audio:
        rel = wav.relative_to(root).as_posix()
        safe_name(rel)
        texts = [p for p in wav.parent.iterdir() if p.is_file() and p.stem == wav.stem and p.suffix.lower() in ('.lab', '.txt', '.textgrid')]
        if not texts:
            raise ValueError('m11_missing_transcript')
        if len(texts) > 1:
            raise ValueError('m11_ambiguous_transcript')
        if not texts[0].stat().st_size:
            raise ValueError('m11_empty_transcript')
        pairs.append(dict(audio=rel, transcript=texts[0].relative_to(root).as_posix()))
    return pairs
