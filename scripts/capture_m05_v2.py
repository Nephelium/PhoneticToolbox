"""M05 isolated original-V2 oracle; never imports phonetic_core or V3 expected."""
import argparse
import ast
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT.parent / 'PhoneticToolbox_v2'
ORIGINAL_PYTHON = Path.home() / 'Miniconda3_broken_backup_20260317_200156/envs/phonetic_311/python.exe'
OUT = ROOT / 'output/validation/m05'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child(destination,manifest_path=None):
    import math
    import cv2
    import numpy as np
    import mediapipe as mp
    import importlib.metadata as metadata
    metric_path = V2 / 'phonetic_toolbox/core/lip/metrics.py'
    spec = importlib.util.spec_from_file_location('original_m05_metrics', metric_path)
    metrics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_module)
    source_path = V2 / 'phonetic_toolbox/gui/widgets/lip_gui.py'
    source = source_path.read_text('utf-8')
    tree = ast.parse(source)
    stabilizer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LandmarkStabilizer')
    gui = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LipGUI')
    methods = [n for n in gui.body if isinstance(n, ast.FunctionDef) and n.name in ('_recognize_video_frames', '_fill_leading_nans', '_build_mesh_neighbors')]
    oracle_class = ast.ClassDef(name='OriginalOffline', bases=[], keywords=[], body=methods, decorator_list=[])

    class Progress:
        def __init__(self, *args): pass
        def wasCanceled(self): return False
        def __getattr__(self, key): return lambda *args: None

    env = dict(np=np, math=math, cv2=cv2, lip_extract=metrics_module.lip_extract,
               QProgressDialog=Progress, QApplication=SimpleNamespace(processEvents=lambda: None),
               Qt=SimpleNamespace(WindowModality=SimpleNamespace(WindowModal=0)))
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), stabilizer, oracle_class], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source_path), 'exec'), env)
    Original = env['OriginalOffline']
    keys = list(metrics_module.lip_extract(np.zeros((478, 2), np.float32)))
    connections = tuple(set(mp.solutions.face_mesh.FACEMESH_TESSELATION) | set(mp.solutions.face_mesh.FACEMESH_CONTOURS))
    neighbors = Original._build_mesh_neighbors(connections, 478)
    fixture = json.loads((manifest_path or OUT / 'inputs/manifest.json').read_text('utf-8'))
    scientific = []
    for case in fixture['cases']:
        for filtering in (False, True):
            oracle = Original()
            oracle._metrics = {key: [] for key in keys}
            oracle.filter_check = SimpleNamespace(isChecked=lambda: filtering)
            oracle._current_min_cutoff = lambda: 15.0
            oracle._mesh_neighbors = neighbors
            raw = [];rgb_hashes=[]
            with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                min_detection_confidence=.5, min_tracking_confidence=.5) as mesh:
                def process(image):
                    rgb_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
                    result = mesh.process(image)
                    points = None
                    if result.multi_face_landmarks:
                        points = np.array([(p.x * image.shape[1], p.y * image.shape[0]) for p in result.multi_face_landmarks[0].landmark], dtype=np.float32)
                    raw.append(points)
                    return result
                oracle._face_mesh = SimpleNamespace(process=process)
                cap = cv2.VideoCapture(str(OUT / 'inputs' / case['video']))
                try:
                    result = oracle._recognize_video_frames(cap, len(case['frames']), fps_hint=case['fps_hint'])
                finally:
                    cap.release()
            assert result['frame_count'] == len(case['frames'])
            scientific.append(dict(case=case['name'], filter=filtering, raw=raw, rgb_hashes=rgb_hashes, **result))
    # Independently generated coordinates exercise negative openness, degenerate face and filter dynamics.
    rng = np.random.default_rng(20260927)
    input_points = rng.uniform(20, 400, (12, 478, 2)).astype(np.float32)
    input_points[1] = input_points[0] + np.float32(.1)
    input_points[2] = input_points[0] + np.float32(.5)
    input_points[4:7, 13, 1] = 250
    input_points[4:7, 14, 1] = 249
    input_points[8] = 0
    analytic = []
    for cutoff in (1., 15., 30.):
        filter_instance = env['LandmarkStabilizer'](min_cutoff_hz=cutoff, neighbor_indices=neighbors)
        points_out = [filter_instance.filter(points, i * .043) for i, points in enumerate(input_points)]
        analytic.append(dict(cutoff=cutoff, filtered=points_out, metrics=[metrics_module.lip_extract(x) for x in points_out]))
    models = {str(p.relative_to(Path(mp.__file__).parent)): digest(p) for p in (Path(mp.__file__).parent / 'modules').rglob('*.tflite') if 'face_' in str(p)}
    source_hashes = {str(p.relative_to(V2)): digest(p) for p in [metric_path, source_path, V2 / 'phonetic_toolbox/services/lip_service.py', V2 / 'phonetic_toolbox/services/io/lip.py', V2 / 'Phonetic_Export/index.html']}
    def clean(value):
        if isinstance(value, np.ndarray): return clean(value.tolist())
        if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value): return None
        return value
    payload = clean(dict(producer=dict(python=sys.version, versions={n: metadata.version(n) for n in ('mediapipe', 'numpy', 'opencv-contrib-python')},
                                       source_hashes=source_hashes, models=models, mode='original AST methods; UI progress stub only'),
                         spec=dict(outer=metrics_module.OUTER_LIP_LANDMARKS, inner=metrics_module.INNER_LIP_LANDMARKS,
                                   face=metrics_module.FACE_OVAL, neighbors=neighbors, keys=keys),
                         scientific=scientific, input_points=input_points, analytic=analytic))
    with gzip.open(destination, 'wt', encoding='utf-8') as stream:
        json.dump(payload, stream, allow_nan=False, separators=(',', ':'))
    print(json.dumps(dict(cases=len(scientific), frames=sum(c['frame_count'] for c in scientific), original_only=True)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=Path)
    parser.add_argument('--manifest',type=Path)
    args = parser.parse_args()
    if args.child:
        if args.child.exists():raise ValueError('Refusing to overwrite original capture')
        child(args.child,args.manifest)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [OUT / f'v2-{i}.json.gz' for i in (1, 2)]
    if any(p.exists() for p in paths):
        raise ValueError('Refusing to overwrite original captures')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    for path in paths:
        subprocess.run([str(ORIGINAL_PYTHON), '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--child', str(path)],
                       cwd=OUT, env=env, check=True, timeout=180)
    def read(path):
        with gzip.open(path, 'rt', encoding='utf-8') as stream: return json.load(stream)
    first, second = map(read, paths)
    assert first == second, 'V2 repeated capture mismatch'
    target = ROOT / 'tests/fixtures/m05/v2.json.gz'
    target.parent.mkdir(exist_ok=True)
    if target.exists(): raise ValueError('Refusing to overwrite frozen expected')
    target.write_bytes(paths[0].read_bytes())
    print('V2 dual-process capture identical; frozen independent expected')


if __name__ == '__main__':
    main()
