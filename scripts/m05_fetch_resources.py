"""Fetch fixed official M05 resources; no package install scripts or runtime CDN."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'frontend/public/m05/mediapipe-0.10.14'
PACKAGE = 'https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-0.10.14.tgz'
MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
IMAGE = 'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/astronaut.png'


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read(50_000_001)
    if len(payload) > 50_000_000:
        raise ValueError('resource_size_limit')
    return payload


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f'Refusing changed resource: {path.name}')
    if not path.exists():
        path.write_bytes(data)
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    archive = fetch(PACKAGE)
    files = {}
    allowed = {'vision_bundle.cjs', 'vision_bundle.mjs', 'wasm/vision_wasm_internal.js',
               'wasm/vision_wasm_internal.wasm', 'wasm/vision_wasm_nosimd_internal.js',
               'wasm/vision_wasm_nosimd_internal.wasm'}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        for member in tar.getmembers():
            rel = member.name.removeprefix('package/')
            if rel in allowed or rel in ('package.json', 'LICENSE'):
                if not member.isfile() or member.size > 25_000_000:
                    raise ValueError('invalid_archive_member')
                data = tar.extractfile(member).read()
                files[rel] = save(DEST / rel, data)
    if not allowed.issubset(files):
        raise ValueError('missing_runtime_files')
    files['vision_bundle.classic.js'] = save(DEST / 'vision_bundle.classic.js', (DEST / 'vision_bundle.cjs').read_bytes())
    files['face_landmarker.task'] = save(DEST / 'face_landmarker.task', fetch(MODEL))
    image_info = save(ROOT / 'output/validation/m05/inputs/astronaut.png', fetch(IMAGE))
    manifest = dict(schema='m05-resources/1', package_version='0.10.14', package_url=PACKAGE,
                    package_sha256=hashlib.sha256(archive).hexdigest(), model_url=MODEL,
                    model_revision='float16/1', files=files,
                    fixture=dict(url=IMAGE, **image_info,
                                 attribution='NASA / Eileen Collins portrait via scikit-image 0.19.3',
                                 use='Local engineering tests; no endorsement; no commercial likeness use',
                                 source='https://scikit-image.org/docs/0.21.x/api/skimage.data.html#skimage.data.astronaut'))
    manifest['aliases'] = {'vision_bundle.classic.js': 'byte-identical vision_bundle.cjs with JS MIME extension for Qt Assets'}
    path = ROOT / 'resources/m05/resources.json'
    if path.exists():
        if json.loads(path.read_text('utf8')) != manifest: raise ValueError('Resource manifest changed')
    else: save(path, (json.dumps(manifest, indent=2) + '\n').encode())
    print(json.dumps(dict(resource_bytes=sum(v['bytes'] for v in files.values()), manifest=str(path))))


if __name__ == '__main__':
    main()
