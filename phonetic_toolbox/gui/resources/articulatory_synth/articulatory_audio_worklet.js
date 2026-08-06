class ArticulatorySynthCore {
    constructor(sampleRate) {
        this.sampleRate = sampleRate;
        this.phase = 0;
        this.noiseLast = 0;
        this.tiltLast = 0;
        this.radiationLastIn = 0;
        this.radiationLastOut = 0;
        this.nasalLast = 0;
        this.env = 0;
        this.filters = Array.from({ length: 5 }, () => ({ y1: 0, y2: 0 }));
        this.state = {
            isPlaying: false,
            voiced: true,
            f0: 140,
            airflow: 0.62,
            openQuotient: 0.58,
            tongueFront: 0.52,
            tongueHigh: 0.42,
            tongueBody: 0.52,
            jawOpen: 0.45,
            pharynxWidth: 0.55,
            velumHeight: 0.9,
            lipOpen: 0.56,
            lipRound: 0.14,
            nasalOpen: 0,
        };
    }

    setState(nextState) {
        this.state = { ...this.state, ...nextState };
    }

    clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    estimateFormants(values) {
        const front = values.tongueFront;
        const high = values.tongueHigh;
        const lipRound = values.lipRound;
        const lipOpen = values.lipOpen;
        const nasal = values.nasalOpen;
        const jaw = values.jawOpen;
        const pharynx = values.pharynxWidth;
        const f1 = this.clamp(900 - 640 * high + 240 * jaw + 80 * lipOpen + 70 * (1 - pharynx), 160, 1150);
        const f2 = this.clamp(780 + 1660 * front - 390 * lipRound - 150 * high - 90 * (1 - pharynx), 560, 2900);
        const f3 = this.clamp(2450 + 420 * front - 330 * lipRound + 70 * lipOpen + 120 * pharynx - 130 * nasal, 1750, 3700);
        return [f1, f2, f3];
    }

    nextSample() {
        const s = this.state;
        const targetEnv = s.isPlaying ? 1 : 0;
        this.env += (targetEnv - this.env) * 0.0018;
        if (this.env < 0.0001 && !s.isPlaying) {
            return 0;
        }

        const f0 = this.clamp(s.f0 || 140, 30, 600);
        this.phase += f0 / this.sampleRate;
        if (this.phase >= 1) {
            this.phase -= 1;
        }

        const openQ = this.clamp(s.openQuotient || 0.58, 0.12, 0.92);
        let voiced = 0;
        if (s.voiced) {
            if (this.phase < openQ) {
                const p = this.phase / openQ;
                voiced = 0.5 - 0.5 * Math.cos(2 * Math.PI * p);
            } else {
                const p = (this.phase - openQ) / Math.max(0.001, 1 - openQ);
                voiced = -0.18 * Math.sin(Math.PI * p);
            }
            voiced -= 0.12;
            this.tiltLast = 0.965 * this.tiltLast + 0.035 * voiced;
            voiced = this.tiltLast;
        }

        const white = Math.random() * 2 - 1;
        const constriction = this.clamp(s.tongueHigh * s.tongueBody * (1.25 - s.lipOpen - 0.25 * s.jawOpen), 0, 1);
        const frication = white - this.noiseLast * 0.72;
        this.noiseLast = white;
        const noiseGain = s.airflow * (s.voiced ? 0.06 : 0.46) + constriction * s.airflow * 0.34;
        const source = voiced * (s.voiced ? s.airflow * 1.25 : 0) + frication * noiseGain;

        const formants = this.estimateFormants(s);
        const round = s.lipRound || 0;
        const bandwidths = [
            58 + 80 * s.nasalOpen,
            92 + 35 * round,
            132 + 70 * round,
            210,
            280,
        ];
        const freqs = [
            formants[0],
            formants[1],
            formants[2],
            3600 - 260 * round,
            4500 - 320 * round,
        ];
        const weights = [1.0, 0.72, 0.46, 0.22, 0.14];
        let tract = source * 0.05;
        for (let i = 0; i < this.filters.length; i += 1) {
            tract += this.applyResonator(i, source, freqs[i], bandwidths[i]) * weights[i];
        }
        let x = tract * 2.8;

        const nasal = this.clamp(s.nasalOpen || 0, 0, 1);
        if (nasal > 0.001) {
            this.nasalLast = 0.94 * this.nasalLast + 0.06 * x;
            x = x * (1 - nasal * 0.28) + this.nasalLast * nasal * 0.34;
        }

        const radiation = x - this.radiationLastIn + 0.985 * this.radiationLastOut;
        this.radiationLastIn = x;
        this.radiationLastOut = radiation;
        return Math.tanh(radiation * 7.5) * this.env * 0.72;
    }

    applyResonator(index, input, frequency, bandwidth) {
        const f = this.clamp(frequency, 80, this.sampleRate * 0.45);
        const bw = this.clamp(bandwidth, 20, 900);
        const r = Math.exp(-Math.PI * bw / this.sampleRate);
        const c = 2 * r * Math.cos(2 * Math.PI * f / this.sampleRate);
        const gain = 1 - r;
        const filter = this.filters[index];
        const y = gain * input + c * filter.y1 - r * r * filter.y2;
        filter.y2 = filter.y1;
        filter.y1 = y;
        return y;
    }
}

class ArticulatoryProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.core = new ArticulatorySynthCore(sampleRate);
        this.port.onmessage = (event) => {
            if (event.data && event.data.type === "state") {
                this.core.setState(event.data.state);
            }
        };
    }

    process(_inputs, outputs) {
        const output = outputs[0][0];
        for (let i = 0; i < output.length; i += 1) {
            output[i] = this.core.nextSample();
        }
        return true;
    }
}

registerProcessor("articulatory-processor", ArticulatoryProcessor);
