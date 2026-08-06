(function () {
    "use strict";

    const sliderDefs = [
        ["f0", "F0", 50, 360, 140, 1, "Hz"],
        ["airflow", "气流强度", 0, 1, 0.62, 0.01, ""],
        ["openQuotient", "声门开放比", 0.18, 0.88, 0.58, 0.01, ""],
        ["tongueFront", "舌位前后", 0, 1, 0.52, 0.01, ""],
        ["tongueHigh", "舌位高低", 0, 1, 0.42, 0.01, ""],
        ["tongueBody", "舌体隆起", 0, 1, 0.52, 0.01, ""],
        ["jawOpen", "下巴开合", 0, 1, 0.45, 0.01, ""],
        ["pharynxWidth", "咽腔宽窄", 0, 1, 0.55, 0.01, ""],
        ["velumHeight", "软腭抬高", 0, 1, 0.9, 0.01, ""],
        ["lipOpen", "唇口开度", 0.05, 1, 0.56, 0.01, ""],
        ["lipRound", "唇圆展", 0, 1, 0.14, 0.01, ""],
        ["nasalOpen", "鼻腔开度", 0, 1, 0, 0.01, ""],
    ];

    const presets = {
        neutral: {
            f0: 140, airflow: 0.62, openQuotient: 0.58,
            tongueFront: 0.52, tongueHigh: 0.42, tongueBody: 0.52,
            jawOpen: 0.45, pharynxWidth: 0.55, velumHeight: 0.9,
            lipOpen: 0.56, lipRound: 0.14, nasalOpen: 0,
            voiced: true, isPlaying: false,
        },
        i: {
            f0: 150, airflow: 0.58, openQuotient: 0.55,
            tongueFront: 0.92, tongueHigh: 0.9, tongueBody: 0.82,
            jawOpen: 0.28, pharynxWidth: 0.62, velumHeight: 0.92,
            lipOpen: 0.32, lipRound: 0.04, nasalOpen: 0,
            voiced: true,
        },
        a: {
            f0: 132, airflow: 0.7, openQuotient: 0.62,
            tongueFront: 0.38, tongueHigh: 0.08, tongueBody: 0.22,
            jawOpen: 0.92, pharynxWidth: 0.72, velumHeight: 0.88,
            lipOpen: 0.92, lipRound: 0.02, nasalOpen: 0,
            voiced: true,
        },
        u: {
            f0: 138, airflow: 0.56, openQuotient: 0.56,
            tongueFront: 0.08, tongueHigh: 0.86, tongueBody: 0.78,
            jawOpen: 0.24, pharynxWidth: 0.5, velumHeight: 0.92,
            lipOpen: 0.28, lipRound: 0.92, nasalOpen: 0,
            voiced: true,
        },
        nasal: {
            f0: 140, airflow: 0.6, openQuotient: 0.6,
            tongueFront: 0.52, tongueHigh: 0.38, tongueBody: 0.45,
            jawOpen: 0.48, pharynxWidth: 0.56, velumHeight: 0.18,
            lipOpen: 0.58, lipRound: 0.08, nasalOpen: 0.78,
            voiced: true,
        },
        fricative: {
            f0: 120, airflow: 0.92, openQuotient: 0.32,
            tongueFront: 0.84, tongueHigh: 0.96, tongueBody: 0.9,
            jawOpen: 0.25, pharynxWidth: 0.45, velumHeight: 0.94,
            lipOpen: 0.2, lipRound: 0.02, nasalOpen: 0,
            voiced: false,
        },
        voicedFricative: {
            f0: 125, airflow: 0.82, openQuotient: 0.48,
            tongueFront: 0.76, tongueHigh: 0.88, tongueBody: 0.88,
            jawOpen: 0.3, pharynxWidth: 0.48, velumHeight: 0.92,
            lipOpen: 0.24, lipRound: 0.02, nasalOpen: 0,
            voiced: true,
        },
        whisper: {
            f0: 130, airflow: 0.86, openQuotient: 0.82,
            tongueFront: 0.56, tongueHigh: 0.38, tongueBody: 0.42,
            jawOpen: 0.54, pharynxWidth: 0.7, velumHeight: 0.86,
            lipOpen: 0.62, lipRound: 0.08, nasalOpen: 0,
            voiced: false,
        },
    };

    const state = {
        ...presets.neutral,
        isPlaying: false,
        nasalOn: false,
    };

    const els = {
        sliderBank: document.getElementById("sliderBank"),
        playToggle: document.getElementById("playToggle"),
        voiceToggle: document.getElementById("voiceToggle"),
        nasalToggle: document.getElementById("nasalToggle"),
        audioStatus: document.getElementById("audioStatus"),
        engineLabel: document.getElementById("engineLabel"),
        postureLabel: document.getElementById("postureLabel"),
        formantReadout: document.getElementById("formantReadout"),
        constrictionReadout: document.getElementById("constrictionReadout"),
        lipAreaReadout: document.getElementById("lipAreaReadout"),
        nasalReadout: document.getElementById("nasalReadout"),
        areaCanvas: document.getElementById("areaCanvas"),
        waveCanvas: document.getElementById("waveCanvas"),
        spectrumCanvas: document.getElementById("spectrumCanvas"),
        exportParams: document.getElementById("exportParams"),
        exportWav: document.getElementById("exportWav"),
        tractSvg: document.getElementById("tractSvg"),
        tonguePath: document.getElementById("tonguePath"),
        velumPath: document.getElementById("velumPath"),
        uvulaPath: document.getElementById("uvulaPath"),
        oralCavity: document.getElementById("oralCavity"),
        pharynxSpace: document.getElementById("pharynxSpace"),
        upperLip: document.getElementById("upperLip"),
        lowerLip: document.getElementById("lowerLip"),
        jawPath: document.getElementById("jawPath"),
        upperTeeth: document.getElementById("upperTeeth"),
        lowerTeeth: document.getElementById("lowerTeeth"),
        airPath: document.getElementById("airPath"),
        glottisDot: document.getElementById("glottisDot"),
        dragHandles: Array.from(document.querySelectorAll(".drag-handle")),
    };

    const sliderNodes = new Map();

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function formatValue(def, value) {
        if (def[0] === "f0") {
            return `${Math.round(value)} ${def[6]}`;
        }
        return `${Math.round(value * 100)}%`;
    }

    function setParam(name, value) {
        const def = sliderDefs.find((item) => item[0] === name);
        if (!def) {
            return;
        }
        state[name] = clamp(Number(value), Number(def[2]), Number(def[3]));
        const nodes = sliderNodes.get(name);
        if (nodes) {
            nodes.input.value = String(state[name]);
            nodes.output.value = formatValue(def, state[name]);
        }
        if (name === "nasalOpen") {
            state.nasalOn = state.nasalOpen > 0.05;
            state.velumHeight = clamp(1 - state.nasalOpen, 0, 1);
            const velumNodes = sliderNodes.get("velumHeight");
            if (velumNodes) {
                velumNodes.input.value = String(state.velumHeight);
                const velumDef = sliderDefs.find((item) => item[0] === "velumHeight");
                velumNodes.output.value = formatValue(velumDef, state.velumHeight);
            }
        }
        if (name === "velumHeight") {
            state.nasalOpen = clamp(1 - state.velumHeight, 0, 1);
            const nasalNodes = sliderNodes.get("nasalOpen");
            if (nasalNodes) {
                nasalNodes.input.value = String(state.nasalOpen);
                const nasalDef = sliderDefs.find((item) => item[0] === "nasalOpen");
                nasalNodes.output.value = formatValue(nasalDef, state.nasalOpen);
            }
            state.nasalOn = state.nasalOpen > 0.05;
        }
        updateToggles();
        audioEngine.updateState(state);
    }

    function buildSliders() {
        els.sliderBank.innerHTML = "";
        sliderDefs.forEach((def) => {
            const [name, label, min, max, initial, step] = def;
            state[name] = initial;

            const row = document.createElement("div");
            row.className = "slider-row";

            const labelEl = document.createElement("label");
            labelEl.htmlFor = `slider-${name}`;
            labelEl.textContent = label;

            const output = document.createElement("output");
            output.value = formatValue(def, state[name]);
            labelEl.appendChild(output);

            const input = document.createElement("input");
            input.id = `slider-${name}`;
            input.type = "range";
            input.min = String(min);
            input.max = String(max);
            input.step = String(step);
            input.value = String(state[name]);
            input.addEventListener("input", () => setParam(name, input.value));

            row.appendChild(labelEl);
            row.appendChild(input);
            els.sliderBank.appendChild(row);
            sliderNodes.set(name, { input, output });
        });
    }

    function computeArea(values) {
        const n = 32;
        const areas = [];
        const front = values.tongueFront;
        const high = values.tongueHigh;
        const body = values.tongueBody;
        const jaw = values.jawOpen;
        const pharynxWidth = values.pharynxWidth;
        const lipOpen = values.lipOpen;
        const lipRound = values.lipRound;
        const constrictionCenter = 0.28 + 0.54 * front;
        const constrictionWidth = 0.07 + 0.12 * (1 - body) + 0.04 * jaw;
        const constrictionDepth = 0.35 + 2.35 * high * (0.55 + 0.45 * body) - 0.7 * jaw;
        const tongueRootNarrowing = Math.max(0, 0.7 - front) * (0.35 + 0.75 * high);
        const pharynxNarrowing = (1 - pharynxWidth) * 1.5 + tongueRootNarrowing;
        const lipArea = clamp(0.18 + 1.55 * lipOpen + 1.05 * jaw - 1.05 * lipRound, 0.035, 3.2);

        for (let i = 0; i < n; i += 1) {
            const x = i / (n - 1);
            let area = 1.55 + 0.82 * Math.sin(Math.PI * x) + 0.82 * jaw;
            area -= pharynxNarrowing * Math.exp(-Math.pow((x - 0.14) / 0.16, 2));
            area -= constrictionDepth * Math.exp(-Math.pow((x - constrictionCenter) / constrictionWidth, 2));
            area += 0.34 * (1 - high) * Math.exp(-Math.pow((x - 0.62) / 0.26, 2));
            if (x > 0.84) {
                const mix = (x - 0.84) / 0.16;
                area = area * (1 - mix) + lipArea * mix;
            }
            areas.push(clamp(area, 0.045, 4.0));
        }
        return areas;
    }

    function estimateFormants(values) {
        const front = values.tongueFront;
        const high = values.tongueHigh;
        const lipRound = values.lipRound;
        const lipOpen = values.lipOpen;
        const nasal = values.nasalOpen;
        const jaw = values.jawOpen;
        const pharynx = values.pharynxWidth;
        const f1 = clamp(900 - 640 * high + 240 * jaw + 80 * lipOpen + 70 * (1 - pharynx), 160, 1150);
        const f2 = clamp(780 + 1660 * front - 390 * lipRound - 150 * high - 90 * (1 - pharynx), 560, 2900);
        const f3 = clamp(2450 + 420 * front - 330 * lipRound + 70 * lipOpen + 120 * pharynx - 130 * nasal, 1750, 3700);
        return [f1, f2, f3];
    }

    function describePosture() {
        const height = state.tongueHigh > 0.67 ? "高" : state.tongueHigh < 0.3 ? "低" : "中";
        const front = state.tongueFront > 0.67 ? "前" : state.tongueFront < 0.3 ? "后" : "央";
        const lip = state.lipRound > 0.55 ? "圆唇" : state.lipOpen > 0.72 ? "开口" : "展唇";
        const nasal = state.nasalOpen > 0.1 ? "鼻化" : "口腔";
        return `${height}${front} ${lip} ${nasal}`;
    }

    function updateToggles() {
        els.playToggle.textContent = state.isPlaying ? "停止发声" : "开始发声";
        els.voiceToggle.textContent = `声带：${state.voiced ? "开" : "关"}`;
        els.nasalToggle.textContent = `鼻腔：${state.nasalOpen > 0.05 ? "开" : "关"}`;
        els.voiceToggle.classList.toggle("active-toggle", state.voiced);
        els.nasalToggle.classList.toggle("active-toggle", state.nasalOpen > 0.05);
    }

    function updateSvg() {
        const front = state.tongueFront;
        const high = state.tongueHigh;
        const body = state.tongueBody;
        const jaw = state.jawOpen;
        const pharynx = state.pharynxWidth;
        const velumHeight = state.velumHeight;
        const lipOpen = state.lipOpen;
        const lipRound = state.lipRound;
        const nasal = state.nasalOpen;
        const dorsumX = 255 + 215 * front;
        const dorsumY = 305 - 112 * high;
        const rootY = 328 - 42 * (1 - front) * high + 12 * (1 - pharynx);
        const tipX = 178 + 62 * front;
        const tipY = 314 - 56 * high + 12 * jaw;
        const bladeX = 206 + 98 * front;
        const bladeY = 288 - 78 * high * body + 8 * jaw;
        const backX = 520 - 18 * lipRound - 18 * (1 - pharynx);
        const backY = 342 - 26 * high + 16 * (1 - pharynx);
        const floorY = 382 + 22 * jaw;
        els.tonguePath.setAttribute(
            "d",
            `M145 ${rootY} C166 ${302 - 26 * high + 10 * jaw} ${tipX} ${tipY} ${bladeX} ${bladeY} ` +
            `C${dorsumX - 42} ${dorsumY} ${dorsumX + 56} ${dorsumY - 8 * body} ${backX} ${backY} ` +
            `C430 ${floorY - 18 * body} 258 ${floorY - 8 * high} 145 ${rootY} Z`
        );

        const lipShift = 24 * lipRound;
        const lipGap = 24 + 54 * lipOpen + 40 * jaw;
        els.upperLip.setAttribute(
            "d",
            `M103 ${204 - lipGap * 0.18} C${76 - lipShift} ${188 - lipGap * 0.12} 55 ${193 - lipGap * 0.05} 42 ${214 - lipGap * 0.17} C67 ${214 - lipGap * 0.14} 89 ${218 - lipGap * 0.17} 112 ${231 - lipGap * 0.18}`
        );
        els.lowerLip.setAttribute(
            "d",
            `M112 ${231 + lipGap * 0.22} C${86 - lipShift} ${258 + lipGap * 0.2} 59 ${277 + lipGap * 0.12} 39 ${258 + lipGap * 0.16} C67 ${252 + lipGap * 0.13} 90 ${244 + lipGap * 0.15} 113 ${230 + lipGap * 0.16}`
        );
        els.upperTeeth.setAttribute("d", `M119 ${199 - lipGap * 0.08} L108 ${239 - lipGap * 0.08}`);
        els.lowerTeeth.setAttribute("d", `M115 ${249 + lipGap * 0.18} L105 ${279 + lipGap * 0.18}`);
        els.jawPath.setAttribute(
            "d",
            `M108 ${330 + 30 * jaw} C183 ${397 + 18 * jaw} 366 ${408 + 12 * jaw} 512 ${360 + 5 * jaw}`
        );
        els.oralCavity.setAttribute(
            "d",
            `M111 211 C213 ${168 - 10 * high} 348 ${152 - 14 * high} 492 166 C543 171 575 201 577 246 C551 ${235 + 8 * jaw} 515 ${221 + 12 * jaw} 456 ${215 + 16 * jaw} C332 ${202 + 20 * jaw} 228 ${218 + 22 * jaw} 134 ${273 + 16 * jaw} C101 ${257 + 18 * jaw} 91 231 111 211 Z`
        );
        els.pharynxSpace.setAttribute(
            "d",
            `M${530 + 18 * pharynx} 158 C${597 + 18 * pharynx} 186 ${622 + 18 * pharynx} 242 ${599 + 22 * pharynx} 304 C${585 + 18 * pharynx} 343 551 380 512 405 C514 347 ${519 + 20 * pharynx} 288 ${542 + 24 * pharynx} 242 C${559 + 18 * pharynx} 209 ${555 + 10 * pharynx} 181 ${530 + 18 * pharynx} 158 Z`
        );

        const velumDrop = 64 * (1 - velumHeight);
        els.velumPath.setAttribute(
            "d",
            `M532 169 C568 ${179 + velumDrop * 0.16} 589 ${206 + velumDrop * 0.5} 592 ${246 + velumDrop}`
        );
        els.uvulaPath.setAttribute(
            "d",
            `M572 ${231 + velumDrop * 0.35} C558 ${250 + velumDrop * 0.5} 558 ${268 + velumDrop * 0.52} 573 ${286 + velumDrop * 0.58}`
        );
        els.nasalToggle.classList.toggle("active-toggle", nasal > 0.05);
        els.tractSvg.style.setProperty("--nasal-opacity", String(0.2 + nasal * 0.65));
        document.getElementById("nasalPath").style.opacity = String(0.18 + nasal * 0.58);
        els.glottisDot.style.opacity = state.voiced ? "1" : "0.35";
        els.airPath.style.opacity = String(0.25 + state.airflow * 0.65);
        els.airPath.setAttribute(
            "d",
            `M129 ${234 + lipGap * 0.02} C235 ${190 - 32 * high} ${374} ${183 - 20 * high} 510 ${204 + 8 * nasal} C552 ${211 + 8 * nasal} 573 ${222 + 18 * nasal} 589 ${240 + 24 * nasal}`
        );
        updateDragHandles({
            upperLip: [88 - lipShift * 0.55, 206 - lipGap * 0.16],
            lowerLip: [88 - lipShift * 0.55, 254 + lipGap * 0.2],
            jaw: [168, 354 + 32 * jaw],
            tongueTip: [tipX, tipY],
            tongueBody: [dorsumX, dorsumY],
            tongueRoot: [backX - 20, backY - 8],
            velum: [590, 228 + velumDrop * 0.75],
            pharynx: [563 + 24 * pharynx, 248],
        });
        els.postureLabel.textContent = describePosture();
    }

    function updateDragHandles(points) {
        els.dragHandles.forEach((handle) => {
            const point = points[handle.dataset.control];
            if (!point) {
                return;
            }
            handle.setAttribute("cx", String(point[0]));
            handle.setAttribute("cy", String(point[1]));
        });
    }

    function svgPointFromEvent(event) {
        const point = els.tractSvg.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        return point.matrixTransform(els.tractSvg.getScreenCTM().inverse());
    }

    function applyDrag(control, point) {
        const x = point.x;
        const y = point.y;
        if (control === "upperLip") {
            setParam("lipRound", clamp((104 - x) / 44, 0, 1));
            setParam("lipOpen", clamp((238 - y) / 78, 0.05, 1));
        } else if (control === "lowerLip") {
            setParam("lipRound", clamp((104 - x) / 44, 0, 1));
            setParam("lipOpen", clamp((y - 226) / 88, 0.05, 1));
        } else if (control === "jaw") {
            setParam("jawOpen", clamp((y - 342) / 58, 0, 1));
            setParam("lipOpen", clamp(state.lipOpen + (state.jawOpen - 0.45) * 0.06, 0.05, 1));
        } else if (control === "tongueTip") {
            setParam("tongueFront", clamp((x - 176) / 66, 0, 1));
            setParam("tongueHigh", clamp((320 - y) / 76, 0, 1));
        } else if (control === "tongueBody") {
            setParam("tongueFront", clamp((x - 255) / 215, 0, 1));
            setParam("tongueHigh", clamp((305 - y) / 112, 0, 1));
            setParam("tongueBody", clamp((300 - y) / 116, 0, 1));
        } else if (control === "tongueRoot") {
            setParam("tongueFront", clamp(1 - (y - 286) / 90, 0, 1));
            setParam("pharynxWidth", clamp((x - 510) / 72, 0, 1));
            setParam("tongueBody", clamp((342 - y) / 90, 0, 1));
        } else if (control === "velum") {
            setParam("velumHeight", clamp(1 - (y - 224) / 78, 0, 1));
        } else if (control === "pharynx") {
            setParam("pharynxWidth", clamp((x - 540) / 70, 0, 1));
        }
    }

    function drawArea() {
        const canvas = els.areaCanvas;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;
        const areas = computeArea(state);
        const formants = estimateFormants(state);
        const minArea = Math.min(...areas);
        const lipArea = areas[areas.length - 1];

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#fbfcf8";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "#d9e2da";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 8; i += 1) {
            const y = 22 + i * ((h - 50) / 8);
            ctx.beginPath();
            ctx.moveTo(34, y);
            ctx.lineTo(w - 18, y);
            ctx.stroke();
        }

        ctx.strokeStyle = "#007f72";
        ctx.lineWidth = 4;
        ctx.beginPath();
        areas.forEach((area, index) => {
            const x = 36 + index * ((w - 62) / (areas.length - 1));
            const y = h - 28 - (area / 4.0) * (h - 58);
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();

        ctx.fillStyle = "#171916";
        ctx.font = "14px Microsoft YaHei, sans-serif";
        ctx.fillText("声门", 30, h - 8);
        ctx.fillText("唇端", w - 58, h - 8);
        ctx.fillText("cm²", 8, 22);

        els.formantReadout.textContent = `F1 ${Math.round(formants[0])} / F2 ${Math.round(formants[1])} / F3 ${Math.round(formants[2])} Hz`;
        els.constrictionReadout.textContent = `${minArea.toFixed(2)} cm²`;
        els.lipAreaReadout.textContent = `${lipArea.toFixed(2)} cm²`;
        els.nasalReadout.textContent = `${Math.round(state.nasalOpen * 100)}%`;
    }

    function drawWave() {
        const canvas = els.waveCanvas;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#fbfcf8";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "#d9e2da";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();

        const analyser = audioEngine.analyser;
        if (!analyser) {
            return;
        }
        const data = new Uint8Array(analyser.fftSize);
        analyser.getByteTimeDomainData(data);
        ctx.strokeStyle = "#c73242";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < data.length; i += 1) {
            const x = (i / (data.length - 1)) * w;
            const y = (data[i] / 255) * h;
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }

    function drawSpectrum() {
        const canvas = els.spectrumCanvas;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#fbfcf8";
        ctx.fillRect(0, 0, w, h);

        const analyser = audioEngine.analyser;
        if (!analyser) {
            ctx.fillStyle = "#68706a";
            ctx.fillText("启动音频后显示频谱", 20, 28);
            return;
        }

        const data = new Float32Array(analyser.frequencyBinCount);
        analyser.getFloatFrequencyData(data);
        const nyquist = audioEngine.context ? audioEngine.context.sampleRate / 2 : 22050;
        const maxHz = 5000;
        const maxBin = Math.min(data.length - 1, Math.floor((maxHz / nyquist) * data.length));
        ctx.strokeStyle = "#d9e2da";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i += 1) {
            const x = (i / 5) * w;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }

        ctx.fillStyle = "#007f72";
        for (let i = 0; i < maxBin; i += 1) {
            const db = clamp((data[i] + 95) / 75, 0, 1);
            const x = (i / maxBin) * w;
            const barH = db * (h - 18);
            ctx.fillRect(x, h - barH, Math.max(1, w / maxBin), barH);
        }

        ctx.fillStyle = "#171916";
        ctx.font = "13px Microsoft YaHei, sans-serif";
        ctx.fillText("0 Hz", 8, h - 6);
        ctx.fillText("5 kHz", w - 48, h - 6);
    }

    function render() {
        updateSvg();
        drawArea();
        drawWave();
        drawSpectrum();
        requestAnimationFrame(render);
    }

    class SynthCore {
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
            this.state = { ...state };
        }

        setState(nextState) {
            this.state = { ...this.state, ...nextState };
        }

        nextSample() {
            const s = this.state;
            const targetEnv = s.isPlaying ? 1 : 0;
            this.env += (targetEnv - this.env) * 0.0018;
            if (this.env < 0.0001 && !s.isPlaying) {
                return 0;
            }

            const f0 = clamp(s.f0 || 140, 30, 600);
            this.phase += f0 / this.sampleRate;
            if (this.phase >= 1) {
                this.phase -= 1;
            }

            const openQ = clamp(s.openQuotient || 0.58, 0.12, 0.92);
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
            const constriction = clamp(s.tongueHigh * s.tongueBody * (1.25 - s.lipOpen - 0.25 * s.jawOpen), 0, 1);
            const frication = white - this.noiseLast * 0.72;
            this.noiseLast = white;
            const noiseGain = s.airflow * (s.voiced ? 0.06 : 0.46) + constriction * s.airflow * 0.34;
            const source = voiced * (s.voiced ? s.airflow * 1.25 : 0) + frication * noiseGain;

            const formants = estimateFormants(s);
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

            const nasal = clamp(s.nasalOpen || 0, 0, 1);
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
            const f = clamp(frequency, 80, this.sampleRate * 0.45);
            const bw = clamp(bandwidth, 20, 900);
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

    class AudioEngine {
        constructor() {
            this.context = null;
            this.analyser = null;
            this.sourceNode = null;
            this.recorderNode = null;
            this.masterGain = null;
            this.fallbackCore = null;
            this.recentChunks = [];
            this.recentLength = 0;
            this.maxRecentSeconds = 5;
        }

        async ensureStarted() {
            if (!this.context) {
                const AudioContextClass = window.AudioContext || window.webkitAudioContext;
                this.context = new AudioContextClass();
                this.analyser = this.context.createAnalyser();
                this.analyser.fftSize = 2048;
                this.analyser.smoothingTimeConstant = 0.78;
                await this.createSource();
            }
            if (this.context.state !== "running") {
                await this.context.resume();
            }
        }

        async createSource() {
            try {
                if (!this.context.audioWorklet) {
                    throw new Error("AudioWorklet unavailable");
                }
                await this.context.audioWorklet.addModule("articulatory_audio_worklet.js");
                const node = new AudioWorkletNode(this.context, "articulatory-processor", {
                    numberOfInputs: 0,
                    numberOfOutputs: 1,
                    outputChannelCount: [1],
                });
                this.sourceNode = node;
                this.connectGraph(node);
                els.engineLabel.textContent = "AudioWorklet 实时引擎";
                els.audioStatus.textContent = `引擎就绪 ${Math.round(this.context.sampleRate)} Hz`;
            } catch (error) {
                const node = this.context.createScriptProcessor(512, 0, 1);
                this.fallbackCore = new SynthCore(this.context.sampleRate);
                node.onaudioprocess = (event) => {
                    const output = event.outputBuffer.getChannelData(0);
                    this.fallbackCore.setState(state);
                    for (let i = 0; i < output.length; i += 1) {
                        output[i] = this.fallbackCore.nextSample();
                    }
                };
                this.sourceNode = node;
                this.connectGraph(node);
                els.engineLabel.textContent = "兼容模式实时引擎";
                els.audioStatus.textContent = `兼容模式 ${Math.round(this.context.sampleRate)} Hz`;
            }
            this.updateState(state);
        }

        connectGraph(source) {
            this.masterGain = this.context.createGain();
            this.masterGain.gain.value = 0.85;
            const silentSink = this.context.createGain();
            silentSink.gain.value = 0;
            this.recorderNode = this.context.createScriptProcessor(2048, 1, 1);
            this.recorderNode.onaudioprocess = (event) => {
                const input = event.inputBuffer.getChannelData(0);
                const output = event.outputBuffer.getChannelData(0);
                output.fill(0);
                this.storeChunk(input);
            };
            source.connect(this.masterGain);
            this.masterGain.connect(this.analyser);
            this.analyser.connect(this.context.destination);
            this.masterGain.connect(this.recorderNode);
            this.recorderNode.connect(silentSink);
            silentSink.connect(this.context.destination);
        }

        updateState(nextState) {
            if (this.sourceNode && this.sourceNode.port) {
                this.sourceNode.port.postMessage({ type: "state", state: nextState });
            }
            if (this.fallbackCore) {
                this.fallbackCore.setState(nextState);
            }
        }

        storeChunk(input) {
            if (!this.context) {
                return;
            }
            const copy = new Float32Array(input.length);
            copy.set(input);
            this.recentChunks.push(copy);
            this.recentLength += copy.length;
            const maxLength = Math.floor(this.context.sampleRate * this.maxRecentSeconds);
            while (this.recentLength > maxLength && this.recentChunks.length > 1) {
                const removed = this.recentChunks.shift();
                this.recentLength -= removed.length;
            }
        }

        recentSamples() {
            const out = new Float32Array(this.recentLength);
            let offset = 0;
            this.recentChunks.forEach((chunk) => {
                out.set(chunk, offset);
                offset += chunk.length;
            });
            return out;
        }
    }

    const audioEngine = new AudioEngine();

    function encodeWav(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        function writeString(offset, text) {
            for (let i = 0; i < text.length; i += 1) {
                view.setUint8(offset + i, text.charCodeAt(i));
            }
        }

        writeString(0, "RIFF");
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, "WAVE");
        writeString(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, "data");
        view.setUint32(40, samples.length * 2, true);

        let offset = 44;
        for (let i = 0; i < samples.length; i += 1) {
            const sample = clamp(samples[i], -1, 1);
            view.setInt16(offset, sample < 0 ? sample * 32768 : sample * 32767, true);
            offset += 2;
        }
        return new Blob([view], { type: "audio/wav" });
    }

    function downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1200);
    }

    async function togglePlay() {
        await audioEngine.ensureStarted();
        state.isPlaying = !state.isPlaying;
        audioEngine.updateState(state);
        updateToggles();
        els.audioStatus.textContent = state.isPlaying
            ? `正在发声 ${Math.round(audioEngine.context.sampleRate)} Hz`
            : "已停止";
    }

    function applyPreset(name) {
        const preset = presets[name];
        if (!preset) {
            return;
        }
        Object.entries(preset).forEach(([key, value]) => {
            if (sliderDefs.some((def) => def[0] === key)) {
                setParam(key, value);
            } else {
                state[key] = value;
            }
        });
        state.nasalOn = state.nasalOpen > 0.05;
        updateToggles();
        audioEngine.updateState(state);
    }

    function bindEvents() {
        els.playToggle.addEventListener("click", togglePlay);
        els.voiceToggle.addEventListener("click", () => {
            state.voiced = !state.voiced;
            updateToggles();
            audioEngine.updateState(state);
        });
        els.nasalToggle.addEventListener("click", () => {
            setParam("nasalOpen", state.nasalOpen > 0.05 ? 0 : 0.76);
        });
        document.querySelectorAll("[data-preset]").forEach((button) => {
            button.addEventListener("click", () => applyPreset(button.dataset.preset));
        });
        els.exportParams.addEventListener("click", () => {
            const payload = {
                description: "PhoneticToolbox articulatory synth parameters",
                exportedAt: new Date().toISOString(),
                state: { ...state },
                areaFunctionCm2: computeArea(state),
                estimatedFormantsHz: estimateFormants(state),
            };
            const blob = new Blob([JSON.stringify(payload, null, 2)], {
                type: "application/json",
            });
            downloadBlob(blob, "articulatory_synth_params.json");
        });
        els.exportWav.addEventListener("click", () => {
            if (!audioEngine.context || audioEngine.recentLength === 0) {
                els.audioStatus.textContent = "暂无可导出的音频";
                return;
            }
            const samples = audioEngine.recentSamples();
            const blob = encodeWav(samples, audioEngine.context.sampleRate);
            downloadBlob(blob, "articulatory_synth_recent.wav");
        });

        let activeControl = null;
        els.dragHandles.forEach((handle) => {
            handle.addEventListener("pointerdown", (event) => {
                event.preventDefault();
                activeControl = handle.dataset.control;
                handle.classList.add("active");
                handle.setPointerCapture(event.pointerId);
                applyDrag(activeControl, svgPointFromEvent(event));
            });
            handle.addEventListener("pointermove", (event) => {
                if (!activeControl || !handle.hasPointerCapture(event.pointerId)) {
                    return;
                }
                event.preventDefault();
                applyDrag(activeControl, svgPointFromEvent(event));
            });
            const release = (event) => {
                if (handle.hasPointerCapture(event.pointerId)) {
                    handle.releasePointerCapture(event.pointerId);
                }
                handle.classList.remove("active");
                activeControl = null;
            };
            handle.addEventListener("pointerup", release);
            handle.addEventListener("pointercancel", release);
        });

        window.addEventListener("keydown", async (event) => {
            if (event.repeat || event.target instanceof HTMLInputElement) {
                return;
            }
            const key = event.key.toLowerCase();
            if (key === " ") {
                event.preventDefault();
                await togglePlay();
            } else if (key === "v") {
                state.voiced = !state.voiced;
                updateToggles();
                audioEngine.updateState(state);
            } else if (key === "n") {
                setParam("nasalOpen", state.nasalOpen > 0.05 ? 0 : 0.76);
            } else {
                const step = event.shiftKey ? 0.02 : 0.055;
                const actions = {
                    w: ["tongueHigh", step],
                    s: ["tongueHigh", -step],
                    a: ["tongueFront", -step],
                    d: ["tongueFront", step],
                    q: ["lipOpen", -step],
                    e: ["lipOpen", step],
                    r: ["lipRound", step],
                    f: ["lipRound", -step],
                };
                if (actions[key]) {
                    event.preventDefault();
                    const [name, delta] = actions[key];
                    setParam(name, state[name] + delta);
                }
            }
        });
    }

    buildSliders();
    bindEvents();
    updateToggles();
    render();
}());
