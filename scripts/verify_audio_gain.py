"""Owned verification outputs and disposable synthetic audio/profile/databases."""
import os
from pathlib import Path
import subprocess
import sys
from verification_artifacts import verification_run,write_json
from workbench_source import configure

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    configure()
    with verification_run('audio','automatic-gain','Synthetic in-memory PCM; gain/settings/experiment volume; isolated Chrome and Qt; no microphone or physical output; pytest owns only scratch') as (out,scratch):
        print(out,flush=True)
        if '--python' in sys.argv:
            import pytest
            result=pytest.main(['-c',str(ROOT/'tests/pytest.ini'),str(ROOT/'desktop/tests/test_playback_gain.py'),str(ROOT/'desktop/tests/test_m10_audio_output.py'),str(ROOT/'desktop/tests/test_m16_recording.py'),str(ROOT/'backend/tests/test_vocal_web.py'),str(ROOT/'backend/tests/test_perception_preferences.py'),'--basetemp',str(scratch/'pytest'),'-q','-o','addopts='])
            write_json(out/'report.json',{'success':result==0,'pytest_exit':int(result)})
            if result:raise RuntimeError('Python gain regression failed')
        else:
            result=subprocess.run(['node',str(ROOT/'tests/e2e/audio-gain.cjs')],cwd=ROOT,env=dict(os.environ,AUDIO_GAIN_OUT=str(out),AUDIO_GAIN_SCRATCH=str(scratch)),capture_output=True,text=True,encoding='utf-8',errors='replace')
            (out/'console.log').write_text(result.stdout+result.stderr,encoding='utf-8');print(result.stdout+result.stderr,flush=True)
            if result.returncode:raise RuntimeError('Actual audio UI verification failed')
