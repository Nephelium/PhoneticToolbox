"""Explicit local WASAPI output check. Sends zero-amplitude PCM only."""
import json
from pathlib import Path
import time

from workbench_source import configure
configure()
import numpy as np
import sounddevice as sd
from phonetic_core.vocal_tract.engine import Engine
from ptb_desktop.vocal_tract.audio_output import LiveAudio, output_devices
from verification_artifacts import verification_run

ROOT=Path(__file__).resolve().parents[1]


def main():
    with verification_run('m10-audio','wasapi','LiveAudio volume=0, engine 48000 Hz, all enumerated WASAPI outputs; no capture or persistent settings') as (out,scratch):
        report={'sounddevice':sd.__version__,'amplitude':0,'devices':[]}
        engine=Engine(resource_dir=ROOT/'resources/vocal_tract/native')
        player=LiveAudio(engine,playback_allowed=True)
        try:
            for device in output_devices():
                if not device['id'].startswith('Windows WASAPI|'):continue
                row={'name':device['name'],'default_rate':sd.query_devices(device['index'])['default_samplerate'],'modes':[]}
                report['devices'].append(row)
                try:
                    sd.check_output_settings(device=device['index'],samplerate=engine.sr,channels=device['channels'],dtype='float32')
                    row['previous_format_check']='supported'
                except Exception as exc:row['previous_format_check']=str(exc)
                player.configure(device=device['id'],volume=0)
                for mode in ['test','preview','animation','live']:
                    if mode=='test':player.test_tone()
                    elif mode=='live':player.start();time.sleep(.2);player.stop()
                    else:player.start(np.zeros(4800),mode=mode)
                    if mode!='live':player.thread.join(3)
                    result={'mode':mode,'error':player.error,'completed':player.completed,'stats':dict(player.stats)}
                    row['modes'].append(result)
                    assert not player.thread.is_alive() and player.error is None and player.stats['blocks']>0,result
                    if mode!='live':assert player.completed,result
                    samples,_,_=player.output_history.snapshot(1)
                    assert not np.any(samples)
            assert report['devices']
            report['passed']=True
        finally:
            player.stop();engine.close()
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(out/'report.json')


if __name__=='__main__':main()
