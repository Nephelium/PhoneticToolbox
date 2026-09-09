<script setup lang="ts">
import { ref, shallowRef, computed, onMounted, onBeforeUnmount, nextTick, watch } from 'vue';
import { parseWav, pcmSlice, selection, sampleAt, type AudioData, type Interval } from './audio.ts';

const audio = shallowRef<AudioData | null>(null), name = ref(''), error = ref('');
const range = ref<Interval>([0, 0]), viewport = ref<Interval>([0, 0]);
const dark = ref(false), status = ref('尚未载入音频'), volume = ref(0.18);
const playing = ref(false), paused = ref(false), cursor = ref(0), loading = ref(false);
const fileInput = ref<HTMLInputElement>(), frames = computed(() => range.value[1] - range.value[0]);
const duration = computed(() => audio.value ? audio.value.sampleCount / audio.value.sampleRate : 0);
let context: AudioContext | null = null, source: AudioBufferSourceNode | null = null, gain: GainNode | null = null;
let beganAt = 0, beganSample = 0, playbackEnd = 0, generation = 0, raf = 0;
let observer: ResizeObserver | null = null;
const canvases = new Map<number, HTMLCanvasElement>();
const time = (sample: number) => audio.value ? (sample / audio.value.sampleRate).toFixed(6) : '0.000000';

function stop(reset = true) {
  generation++;
  if (source) { source.onended = null; source.stop(); source.disconnect(); source = null; }
  playing.value = false; paused.value = false;
  if (reset) cursor.value = range.value[0];
  status.value = audio.value ? '就绪' : '尚未载入音频';
}
function choose(start: number, end: number) {
  if (!audio.value) return;
  const next = selection(start, end, audio.value.sampleCount);
  stop(); range.value = next; cursor.value = start; error.value = '';
}
function editRange(which: 0 | 1, event: Event) {
  const input = event.target as HTMLInputElement;
  try { const next: Interval = [...range.value]; next[which] = Number(input.value); choose(...next); }
  catch (e) { error.value = String((e as Error).message); input.value = String(range.value[which]); }
}
function install(buffer: ArrayBuffer, filename: string) {
  const parsed = parseWav(buffer); // Validate first so a failed import preserves the current audio.
  stop(); audio.value = parsed; name.value = filename;
  range.value = [0, parsed.sampleCount]; viewport.value = [0, parsed.sampleCount];
  cursor.value = 0; error.value = ''; status.value = '就绪';
}
async function loadFile(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  loading.value = true;
  try {
    if (file.size > 128 * 1024 * 1024) throw new Error('P01 探针文件上限为 128 MiB。');
    install(await file.arrayBuffer(), file.name);
  } catch (e) { error.value = (e as Error).message; }
  finally { loading.value = false; if (fileInput.value) fileInput.value.value = ''; }
}
async function loadFixture() {
  loading.value = true;
  try { const response = await fetch('./fixture.wav'); if (!response.ok) throw new Error('测试音加载失败。');
    install(await response.arrayBuffer(), '双轨测试音 · 44.1 kHz.wav');
  } catch (e) { error.value = (e as Error).message; }
  finally { loading.value = false; }
}
async function play() {
  if (!audio.value || frames.value === 0 || playing.value) return;
  const data = audio.value;
  const start = paused.value ? cursor.value : range.value[0], end = range.value[1];
  stop(false);
  const token = generation;
  try {
    if (!context) { context = new AudioContext(); gain = context.createGain(); gain.connect(context.destination); }
    await context.resume();
    if (generation !== token || start >= end) return;
    const buffer = context.createBuffer(data.channels.length, end - start, data.sampleRate);
    pcmSlice(data, [start, end]).forEach((channel, i) => buffer.copyToChannel(channel, i));
    source = context.createBufferSource(); source.buffer = buffer; source.connect(gain!);
    gain!.gain.value = volume.value; beganAt = context.currentTime; beganSample = start; playbackEnd = end;
    playing.value = true; paused.value = false; status.value = '播放中'; cursor.value = start;
    source.onended = () => { if (token === generation) { source?.disconnect(); source = null;
      playing.value = false; paused.value = false; cursor.value = end; status.value = '播放完成'; } };
    source.start();
  } catch (e) { stop(); error.value = `音频播放失败：${(e as Error).message}`; }
}
function pause() {
  if (!playing.value || !context || !audio.value) return;
  const at = Math.min(playbackEnd, beganSample + Math.floor((context.currentTime - beganAt) * audio.value.sampleRate));
  stop(false); cursor.value = at; paused.value = true; status.value = '已暂停';
}
function zoomSelection() { if (frames.value > 0) viewport.value = [...range.value]; }
function fit() { if (audio.value) viewport.value = [0, audio.value.sampleCount]; }
function point(event: PointerEvent, canvas: HTMLCanvasElement) {
  const bounds = canvas.getBoundingClientRect();
  return sampleAt(event.clientX - bounds.left - 58, bounds.width - 76, viewport.value);
}
function drag(event: PointerEvent) {
  if (!audio.value || event.button !== 0) return;
  const canvas = event.currentTarget as HTMLCanvasElement, anchor = point(event, canvas);
  canvas.setPointerCapture(event.pointerId); choose(anchor, anchor);
  const move = (e: PointerEvent) => { const at = point(e, canvas); choose(Math.min(anchor, at), Math.max(anchor, at)); };
  const end = () => { canvas.removeEventListener('pointermove', move); canvas.removeEventListener('pointerup', end);
    canvas.removeEventListener('pointercancel', end); };
  canvas.addEventListener('pointermove', move); canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
}
function canvasRef(element: unknown, index: number) {
  if (element instanceof HTMLCanvasElement) { canvases.set(index, element); observer?.observe(element); }
  else { const previous = canvases.get(index); if (previous) observer?.unobserve(previous); canvases.delete(index); }
}
function draw() {
  const data = audio.value;
  if (!data) return;
  const styles = getComputedStyle(document.documentElement);
  const color = (key: string) => styles.getPropertyValue(key).trim();
  for (const [index, canvas] of canvases) {
    const channel = data.channels[index]; if (!channel) continue;
    const width = canvas.clientWidth, height = canvas.clientHeight, ratio = window.devicePixelRatio;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    }
    const ctx = canvas.getContext('2d')!; ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
    const left = 58, right = width - 18, top = 15, bottom = height - 32, mid = (top + bottom) / 2;
    const span = viewport.value[1] - viewport.value[0], px = (i: number) => left + (i - viewport.value[0]) / span * (right - left);
    ctx.font = '12px "Segoe UI", sans-serif'; ctx.fillStyle = color('--muted'); ctx.strokeStyle = color('--border'); ctx.lineWidth = 1;
    for (let tick = 0; tick <= 4; tick++) {
      const x = left + tick / 4 * (right - left); ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
      ctx.textAlign = tick === 4 ? 'right' : 'left'; ctx.fillText(((viewport.value[0] + tick / 4 * span) / data.sampleRate).toFixed(3), x, height - 10);
    }
    ctx.textAlign = 'left'; ctx.fillText('+1', 18, top + 5); ctx.fillText('0', 18, mid + 4); ctx.fillText('−1', 18, bottom);
    ctx.beginPath(); ctx.moveTo(left, mid); ctx.lineTo(right, mid); ctx.stroke();
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, right - left, bottom - top); ctx.clip();
    ctx.fillStyle = color('--selection'); ctx.fillRect(px(range.value[0]), top, px(range.value[1]) - px(range.value[0]), bottom - top);
    ctx.strokeStyle = color(index % 2 === 0 ? '--accent' : '--track2'); ctx.lineWidth = 1; ctx.beginPath();
    for (let x = 0; x < right - left; x++) {
      const a = Math.floor(viewport.value[0] + x / (right - left) * span);
      const b = Math.min(data.sampleCount, Math.max(a + 1, Math.floor(viewport.value[0] + (x + 1) / (right - left) * span)));
      let low = 1, high = -1;
      for (let i = a; i < b; i++) { low = Math.min(low, channel[i]); high = Math.max(high, channel[i]); }
      ctx.moveTo(left + x, mid - high * (bottom - top) / 2); ctx.lineTo(left + x, mid - low * (bottom - top) / 2 + 0.5);
    }
    ctx.stroke(); ctx.strokeStyle = color('--text'); ctx.beginPath(); ctx.moveTo(px(cursor.value), top); ctx.lineTo(px(cursor.value), bottom); ctx.stroke(); ctx.restore();
  }
}
let lastCursor = -1;
function animate() {
  if (playing.value && context && audio.value)
    cursor.value = Math.min(playbackEnd, beganSample + Math.floor((context.currentTime - beganAt) * audio.value.sampleRate));
  if (lastCursor !== cursor.value) { lastCursor = cursor.value; draw(); }
  raf = requestAnimationFrame(animate);
}
function keyboard(event: KeyboardEvent) {
  if (event.code !== 'Space' || (event.target as HTMLElement).closest('input,button,textarea,select')) return;
  event.preventDefault(); if (playing.value) pause(); else void play();
}
watch(volume, value => { if (gain) gain.gain.value = value; });
watch([audio, range, viewport, dark], async () => {
  document.documentElement.dataset.theme = dark.value ? 'dark' : 'light';
  await nextTick(); draw();
}, { deep: false });
function snapshot() {
  const actualCursor = playing.value && context && audio.value
    ? Math.min(playbackEnd, beganSample + Math.floor((context.currentTime - beganAt) * audio.value.sampleRate)) : cursor.value;
  return { name: name.value, sampleRate: audio.value?.sampleRate, sampleCount: audio.value?.sampleCount,
    channels: audio.value?.channels.length, selection: [...range.value], viewport: [...viewport.value],
    cursor: actualCursor, playing: playing.value, paused: paused.value, error: error.value,
    contextTime: context?.currentTime, beganAt,
    contextState: context?.state, outputRate: context?.sampleRate, dark: dark.value,
    canvases: [...canvases.values()].map(c => ({ width: c.clientWidth, height: c.clientHeight, pixels: c.width })),
    overflow: document.documentElement.scrollWidth > innerWidth,
    ipaFontLoaded: document.fonts.check('24px "Doulos SIL"') };
}
onMounted(async () => {
  observer = new ResizeObserver(draw); canvases.forEach(c => observer!.observe(c));
  window.addEventListener('keydown', keyboard); raf = requestAnimationFrame(animate);
  if (new URLSearchParams(location.search).has('probe')) {
    (window as any).__probe = { loadFixture, loadBytes: (bytes: number[], filename: string) => install(new Uint8Array(bytes).buffer, filename),
      select: choose, zoomSelection, fit, play, pause, stop, snapshot,
      setTheme: (value: boolean) => { dark.value = value; },
      async offlineCheck() {
        if (!audio.value) throw new Error('No audio');
        const sliced = pcmSlice(audio.value, range.value);
        const offline = new OfflineAudioContext(sliced.length, frames.value, audio.value.sampleRate);
        const buffer = offline.createBuffer(sliced.length, frames.value, audio.value.sampleRate);
        sliced.forEach((channel, i) => buffer.copyToChannel(channel, i));
        const node = offline.createBufferSource(); node.buffer = buffer; node.connect(offline.destination); node.start();
        const output = await offline.startRendering(); let maxError = 0;
        sliced.forEach((channel, c) => { const rendered = output.getChannelData(c);
          for (let i = 0; i < channel.length; i++) maxError = Math.max(maxError, Math.abs(channel[i] - rendered[i])); });
        return { frames: output.length, sampleRate: output.sampleRate, channels: output.numberOfChannels, maxError };
      } };
  }
  await document.fonts.ready;
});
onBeforeUnmount(() => { stop(); void context?.close(); observer?.disconnect(); cancelAnimationFrame(raf); window.removeEventListener('keydown', keyboard); });
</script>

<template>
  <div class="workbench">
    <aside>
      <div class="brand"><img src="/icon.png" alt="波形团子"><div>PhoneticToolbox<small>语音研究工具箱</small></div></div>
      <div class="section-label">P01 · 技术验证</div>
      <div class="nav-selected">≋ <span>音频工作台</span></div>
      <p class="aside-note">桌面宿主与共用界面原型<br>业务模块尚未迁移</p>
      <div class="aside-bottom"><span class="local-dot"></span> 本地运行 · 无需登录</div>
    </aside>
    <main>
      <header><div><span class="eyebrow">原型验证 / 音频与选区</span><h1>听得见，也选得准。</h1></div>
        <button id="theme" @click="dark = !dark">{{ dark ? '☀ 浅色' : '☾ 深色' }}</button></header>
      <div class="toolbar">
        <button id="open" class="primary" :disabled="loading" @click="fileInput?.click()">打开 WAV</button>
        <input ref="fileInput" type="file" accept=".wav,audio/wav" hidden @change="loadFile">
        <button id="fixture" :disabled="loading" @click="loadFixture">载入双轨测试音</button>
        <span class="toolbar-help">{{ loading ? '正在载入…' : '只读载入 · 原始采样率' }}</span>
      </div>
      <div v-if="error" class="error" role="alert">{{ error }}</div>
      <section v-if="!audio" class="empty">
        <div class="empty-wave">∿</div><h2>从一段声音开始</h2>
        <p>打开 WAV，或载入两声道合成测试音。<br>观察波形、拖动选区，再试听选中的片段。</p>
        <span class="format-note">P01：PCM 16 / 24 / 32 位、float32 WAV · 最多 8 声道 · 128 MiB</span>
      </section>
      <section v-else class="audio-panel">
        <div class="file-heading"><div><strong>{{ name }}</strong><small>{{ audio.sampleRate.toLocaleString() }} Hz · {{ audio.channels.length }} 声道 · {{ audio.bits }} bit · {{ audio.sampleCount.toLocaleString() }} 采样帧</small></div><span class="duration">{{ duration.toFixed(3) }} s</span></div>
        <div class="view-tools"><span>波形 · 幅度 [FS] / 时间 [s]</span><div><button id="zoom" :disabled="frames === 0" @click="zoomSelection">缩放到选区</button><button id="fit" @click="fit">显示全部</button></div></div>
        <div v-for="(_, index) in audio.channels" :key="index" class="track">
          <div class="track-label"><span :class="{ alternate: index % 2 === 1 }"></span>声道 {{ index + 1 }}<small>拖动波形选择</small></div>
          <canvas :ref="el => canvasRef(el, index)" @pointerdown="drag" :aria-label="`声道 ${index + 1} 波形，选区也可通过下方采样点输入框调整`"></canvas>
        </div>
        <div class="selection-bar"><label>起点 <input id="start" type="number" min="0" :max="audio.sampleCount" step="1" :value="range[0]" @change="editRange(0, $event)"> samples</label>
          <label>终点 <input id="end" type="number" min="0" :max="audio.sampleCount" step="1" :value="range[1]" @change="editRange(1, $event)"> samples</label>
          <span class="selection-meta">{{ frames.toLocaleString() }} 帧 · {{ (frames / audio.sampleRate).toFixed(6) }} s</span></div>
        <div class="transport"><button id="play" class="primary" :disabled="frames === 0 || playing" @click="play">▶ {{ paused ? '继续播放' : '播放选区' }}</button>
          <button id="pause" :disabled="!playing" @click="pause">Ⅱ 暂停</button><button id="stop" @click="stop()">■ 停止</button>
          <span class="position">{{ time(cursor) }} s</span><label class="volume">音量 <input type="range" min="0" max="1" step="0.01" v-model.number="volume"></label></div>
      </section>
      <section class="ipa-panel"><div><span class="eyebrow">中文与国际音标</span><p class="ipa" lang="und-fonipa">[pʰaː tɕʰiŋ˨˩˦ ɹ̩ ɚ n̩ ã ɡ͡b]</p></div><p>原始采样点决定选区<br>切换主题与缩放保留当前状态</p></section>
      <footer><span><i :class="{ active: playing }"></i>{{ status }}</span><span>选区为 [起点, 终点) · 空格播放 / 暂停</span><span>P01 原型 · 0.0.1</span></footer>
    </main>
  </div>
</template>
