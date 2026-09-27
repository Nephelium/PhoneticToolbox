"""Create only missing public synthetic inputs for the M08 formal-host checks."""
from pathlib import Path
import shutil
import subprocess
import numpy as np
import parselmouth

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'output/validation/m08-wiring';out.mkdir(parents=True,exist_ok=True)
    source=out/'input.wav'
    if not source.exists():
        sound=parselmouth.Sound(.2*np.sin(2*np.pi*150*np.arange(16000)/16000),16000)
        sound.save(str(source),'WAV')
    for ext in ('mp3','flac'):
        target=out/('input.'+ext)
        if target.exists():continue
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:raise RuntimeError('Existing ffmpeg required only to prepare compressed test fixtures; no installation performed')
        subprocess.run([ffmpeg,'-nostdin','-v','error','-n','-i',str(source),str(target)],check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))


if __name__=='__main__':main()
