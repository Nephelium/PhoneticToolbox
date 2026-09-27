from pathlib import Path


ENCODE_PREFIX = "ptbx_"


def encode_text(text: str, prefix: str = ENCODE_PREFIX) -> str:
    return f"{prefix}{text.encode('utf-8').hex()}"


def decode_text(text: str, prefix: str = ENCODE_PREFIX) -> str:
    if not text.startswith(prefix):
        return text
    payload = text[len(prefix):]
    if len(payload) % 2 != 0:
        return text
    try:
        return bytes.fromhex(payload).decode("utf-8")
    except ValueError:
        return text


def split_name_and_suffix(name: str) -> tuple[str, str]:
    suffixes = Path(name).suffixes
    if not suffixes:
        return name, ""
    suffix = "".join(suffixes)
    stem = name[: -len(suffix)]
    return stem, suffix


def encode_fs_name(name: str) -> str:
    stem, suffix = split_name_and_suffix(name)
    return f"{encode_text(stem)}{suffix}"


def decode_fs_name(name: str) -> str:
    stem, suffix = split_name_and_suffix(name)
    return f"{decode_text(stem)}{suffix}"
