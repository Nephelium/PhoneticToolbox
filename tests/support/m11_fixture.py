"""Public deterministic formant-like /a/ pulses. No human recording or ASR."""
import math
from pathlib import Path
import struct
import wave


def make_fixture(root, *, syllables=4):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rate = 16000
    samples = []
    for index in range(int((.2 + syllables * .6 + .2) * rate)):
        t = index / rate - .2
        local = t % .6
        value = 0.
        if 0 <= t < syllables * .6 and local < .48:
            envelope = min(1., local/.04, (.48-local)/.04)
            for harmonic in range(1, 31):
                f = harmonic * 180.
                weight = sum(math.exp(-.5*((f-center)/width)**2) for center,width in [(750.,100.), (1200.,140.), (2600.,200.)]) / harmonic
                value += weight * math.sin(2*math.pi*f*t)
            value *= envelope * .75
        samples.append(max(-32767, min(32767, round(value * 32767))))
    audio = root / '公开 合成.wav'
    with wave.open(str(audio), 'wb') as stream:
        stream.setparams((1,2,rate,len(samples),'NONE','not compressed'))
        stream.writeframes(struct.pack('<'+'h'*len(samples), *samples))
    text = root / '公开 合成.lab'
    text.write_text(' '.join(['a']*syllables), encoding='utf8')
    return audio, text
