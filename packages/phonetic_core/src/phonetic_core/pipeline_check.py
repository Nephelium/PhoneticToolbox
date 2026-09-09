"""Original deterministic execution probe; not a phonetic analysis algorithm."""
from hashlib import sha256


def blocks(sample_count: int, seed: int):
    if type(sample_count) is not int or not 1 <= sample_count <= 4096:
        raise ValueError('Invalid probe sample count')
    if type(seed) is not int or not 0 <= seed <= 255:
        raise ValueError('Invalid probe seed')
    digest = sha256()
    for start in range(0, sample_count, 128):
        end = min(sample_count, start + 128)
        digest.update(bytes((seed + i) % 256 for i in range(start, end)))
        yield {'progress': end / sample_count}
    yield {'result': {'sha256': digest.hexdigest(), 'sample_count': sample_count}}
