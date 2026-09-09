"""P07 byte policy. All amounts are decimal bytes, never display-rounded GB."""
import re

QUOTA_BYTES = 5_000_000_000
RETENTION_SECONDS = 7 * 24 * 3600
TEMP_SECONDS = 24 * 3600
CHUNK_BYTES = 256 * 1024


class StorageError(Exception):
    def __init__(self, code, status=409):
        self.code, self.status = code, status
        super().__init__(code)


def reserve(used, reserved, amount):
    if any(type(v) is not int or v < 0 for v in (used, reserved, amount)):
        raise StorageError('invalid_budget', 422)
    if used + reserved + amount > QUOTA_BYTES:
        raise StorageError('quota_exceeded', 413)
    return reserved + amount


def settle(used, reserved, amount):
    if any(type(v) is not int or v < 0 for v in (used, reserved, amount)) or amount > reserved:
        raise StorageError('storage_inconsistent', 503)
    return used + amount, reserved - amount


def expiry(now, input_expiries=()):
    result = min([now + RETENTION_SECONDS, *input_expiries])
    if result <= now:
        raise StorageError('asset_expired', 410)
    return result


def content_range(header, size):
    if header is None:
        return 0, size, False
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
    if not match or not size:
        raise StorageError('range_not_satisfiable', 416)
    left, right = match.groups()
    if not left and right and int(right) > 0:
        return max(0, size - int(right)), size, True
    if not left:
        raise StorageError('range_not_satisfiable', 416)
    start = int(left)
    end = min(size, int(right) + 1) if right else size
    if start >= size or end <= start:
        raise StorageError('range_not_satisfiable', 416)
    return start, end, True
