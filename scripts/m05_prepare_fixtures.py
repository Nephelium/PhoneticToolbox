"""Public NASA portrait transformations: engineering coverage, not natural speech."""
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/validation/m05/inputs'


def main():
    image = cv2.imread(str(OUT / 'astronaut.png'))
    if image is None:
        raise ValueError('Run m05_fetch_resources.py first')
    base = image[20:280, 120:380]
    cases = []
    for name, size, variable in [('front', 256, False), ('motion-occlusion-vfr', 256, True), ('resolution', 640, False)]:
        directory = OUT / name
        directory.mkdir(exist_ok=True)
        video = directory / 'input.mkv'
        if video.exists():
            raise ValueError('Existing fixture: keep it; use existing manifest or a new output directory')
        frames = []
        pts = 0
        concat = []
        for index in range(30):
                frame = cv2.resize(base, (size, size), interpolation=cv2.INTER_LINEAR)
                label = 'front-static'
                if variable:
                    if index < 3 or 15 <= index < 19:
                        frame[:] = 0
                        label = 'blank-detection-loss'
                    elif 10 <= index < 15:
                        frame[size // 3:size * 2 // 3] = 0
                        label = 'synthetic-occlusion'
                    else:
                        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), (index % 7 - 3) * 4, 1)
                        matrix[:, 2] += (index % 5 - 2) * size / 50
                        frame = cv2.warpAffine(frame, matrix, (size, size))
                        label = 'synthetic-2d-motion-not-profile'
                png = directory / f'{index:04d}.png'
                cv2.imwrite(str(png), frame)
                duration = [40, 80, 120][index % 3] if variable else 40
                concat.extend([f"file '{png.name}'", f'duration {duration / 1000}'])
                frames.append(dict(index=index, time_s=pts / 1000, file=png.name,
                                   sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
                                   rgb_sha256=hashlib.sha256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()).hexdigest(), label=label))
                pts += duration
        listing = directory / 'concat.txt'
        listing.write_text('\n'.join(concat)+'\n', encoding='utf-8')
        subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-f','concat','-safe','1','-i',str(listing),
                        '-fps_mode','vfr','-c:v','ffv1','-pix_fmt','bgr0',str(video)], check=True, timeout=60,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time',
                                '-of','json',str(video)], capture_output=True, check=True, timeout=60,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        decoded = json.loads(probe.stdout)['frames']
        assert len(decoded) == len(frames)
        cap = cv2.VideoCapture(str(video))
        try:
            for row, item in zip(frames, decoded):
                assert abs(float(item['best_effort_timestamp_time']) - row['time_s']) < 1e-9
                ok, frame = cap.read()
                assert ok and hashlib.sha256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() == row['rgb_sha256']
        finally:
            cap.release()
        cases.append(dict(name=name, width=size, height=size, fps_hint=25, variable_rate=variable,
                          video=f'{name}/input.mkv', video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(), frames=frames))
    (OUT / 'manifest.json').write_text(json.dumps(dict(schema='m05-fixtures/1', source='resources/m05/resources.json',
                                                     limitations=['not natural speech', 'not true profile', 'not real camera'],
                                                     cases=cases), indent=2), encoding='utf-8')
    print(f'{len(cases)} videos / {sum(len(c["frames"]) for c in cases)} decoded frames, PNG/RGB/PTS verified')


if __name__ == '__main__':
    main()
