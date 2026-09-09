export interface AudioData { sampleRate: number; sampleCount: number; channels: Float32Array<ArrayBuffer>[]; bits: number }
export type Interval = [number, number];

export function selection(start: number, end: number, count: number): Interval {
  if (![start, end, count].every(Number.isSafeInteger) || start < 0 || end < start || end > count)
    throw new Error('选区必须是音频范围内的整数采样点，且起点不能晚于终点。');
  return [start, end];
}
export function sampleAt(x: number, width: number, view: Interval): number {
  if (!(width > 0) || !Number.isFinite(x)) throw new Error('无效的视图尺寸。');
  return Math.round(view[0] + Math.max(0, Math.min(1, x / width)) * (view[1] - view[0]));
}
export function pcmSlice(audio: AudioData, range: Interval): Float32Array<ArrayBuffer>[] {
  const [start, end] = selection(...range, audio.sampleCount);
  return audio.channels.map(channel => channel.slice(start, end));
}
export function parseWav(buffer: ArrayBuffer): AudioData {
  if (buffer.byteLength < 44 || buffer.byteLength > 128 * 1024 * 1024)
    throw new Error('P01 探针只接受完整、大小不超过 128 MiB 的 WAV。');
  const data = new DataView(buffer);
  const tag = (offset: number) => String.fromCharCode(...new Uint8Array(buffer, offset, 4));
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('请选择 RIFF/WAVE 音频。');
  const limit = data.getUint32(4, true) + 8;
  if (limit !== buffer.byteLength) throw new Error('WAV 文件长度与 RIFF 声明不一致。');
  let fmt = -1, body = -1, bodySize = 0;
  for (let offset = 12; offset + 8 <= limit;) {
    const size = data.getUint32(offset + 4, true), start = offset + 8;
    if (start + size > limit) throw new Error('WAV 数据块不完整。');
    if (tag(offset) === 'fmt ') {
      if (fmt !== -1 || size < 16) throw new Error('WAV 格式块无效。');
      fmt = start;
    }
    if (tag(offset) === 'data') {
      if (body !== -1) throw new Error('P01 暂不支持多个 WAV 数据块。');
      body = start; bodySize = size;
    }
    offset = start + size + (size % 2);
  }
  if (fmt < 0 || body < 0) throw new Error('WAV 缺少格式或音频数据块。');
  const format = data.getUint16(fmt, true), count = data.getUint16(fmt + 2, true);
  const sampleRate = data.getUint32(fmt + 4, true), align = data.getUint16(fmt + 12, true);
  const bits = data.getUint16(fmt + 14, true), bytes = bits / 8;
  if (!((format === 1 && [16, 24, 32].includes(bits)) || (format === 3 && bits === 32)))
    throw new Error('P01 支持 PCM 16/24/32 位或 float32 WAV；其他格式尚未实现。');
  if (count < 1 || count > 8 || sampleRate < 8000 || sampleRate > 192000 || align !== count * bytes
      || data.getUint32(fmt + 8, true) !== sampleRate * align || bodySize % align !== 0 || bodySize === 0)
    throw new Error('WAV 的采样率、声道、帧长或数据长度无效。');
  const sampleCount = bodySize / align;
  const channels = Array.from({ length: count }, () => new Float32Array(sampleCount));
  for (let i = 0; i < sampleCount; i++) for (let c = 0; c < count; c++) {
    const offset = body + i * align + c * bytes;
    let value: number;
    if (format === 3) value = data.getFloat32(offset, true);
    else if (bits === 16) value = data.getInt16(offset, true) / 32768;
    else if (bits === 32) value = data.getInt32(offset, true) / 2147483648;
    else value = ((data.getUint8(offset) | (data.getUint8(offset + 1) << 8)
      | (data.getUint8(offset + 2) << 16)) << 8 >> 8) / 8388608;
    if (!Number.isFinite(value)) throw new Error('WAV 包含非有限采样值。');
    channels[c][i] = value;
  }
  return { sampleRate, sampleCount, channels, bits };
}
