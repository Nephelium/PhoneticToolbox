"""Seed an explicit cache preference in a caller-owned disposable QA profile."""
import json
from pathlib import Path


def seed_portable_profile(profile):
    profile = Path(profile).absolute()
    directory = profile / 'PhoneticToolbox-cache'
    config = profile / 'PhoneticToolbox/v3/portable-cache.json'
    if config.exists():
        value = json.loads(config.read_text('utf8'))
        if value.get('schema') != 'ptb-portable-cache/1' or value.get('directory') != str(directory):
            raise ValueError('QA cache preference is not owned by this fixture')
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'cache-directory.json').write_text('{"schema":"ptb-portable-cache/1"}', encoding='utf8')
    config.parent.mkdir(parents=True, exist_ok=True)
    with config.open('x', encoding='utf8') as stream:
        json.dump({'schema':'ptb-portable-cache/1','directory':str(directory),'previousDirectories':[]}, stream)
    return directory
