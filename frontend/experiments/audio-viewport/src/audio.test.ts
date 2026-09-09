import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseWav, selection, sampleAt, pcmSlice } from './audio.ts';

// Fixed, independently specified PCM words: two channels, three frames at 44.1 kHz.
function wav(format = 1, bits = 16, samples = [0, 32767, -32768, 16384, 8192, -8192]) {
  const data = new ArrayBuffer(44 + samples.length * bits / 8);
  const bytes = new DataView(data);
  for (const [offset, text] of [[0, 'RIFF'], [8, 'WAVE'], [12, 'fmt '], [36, 'data']] as const)
    for (let i = 0; i < text.length; i++) bytes.setUint8(offset + i, text.charCodeAt(i));
  bytes.setUint32(4, data.byteLength - 8, true); bytes.setUint32(16, 16, true);
  bytes.setUint16(20, format, true); bytes.setUint16(22, 2, true);
  bytes.setUint32(24, 44100, true); bytes.setUint32(28, 44100 * bits / 4, true);
  bytes.setUint16(32, bits / 4, true); bytes.setUint16(34, bits, true);
  bytes.setUint32(40, samples.length * bits / 8, true);
  samples.forEach((value, i) => {
    const offset = 44 + i * bits / 8;
    if (format === 3) bytes.setFloat32(offset, value, true);
    else if (bits === 16) bytes.setInt16(offset, value, true);
    else if (bits === 32) bytes.setInt32(offset, value, true);
    else { bytes.setUint8(offset, value & 255); bytes.setUint8(offset + 1, (value >> 8) & 255); bytes.setUint8(offset + 2, (value >> 16) & 255); }
  });
  return data;
}

test('preserves original 44100 Hz, frame count and channel identity', () => {
  const audio = parseWav(wav());
  assert.equal(audio.sampleRate, 44100); assert.equal(audio.sampleCount, 3);
  assert.deepEqual(Array.from(audio.channels[0]), [0, -1, 0.25]);
  assert.deepEqual(Array.from(audio.channels[1]), [32767 / 32768, 0.5, -0.25]);
});
test('single-frame selection at EOF retains exactly that frame', () => {
  assert.deepEqual(pcmSlice(parseWav(wav()), [2, 3]).map(v => Array.from(v)), [[0.25], [-0.25]]);
});
test('zero-length selection stays empty and never becomes play-all', () => {
  assert.equal(pcmSlice(parseWav(wav()), [2, 2])[0].length, 0);
});
test('reversed, outside, noninteger and NaN selections fail explicitly', () => {
  for (const [a, b] of [[2, 1], [-1, 2], [0, 4], [0.5, 2], [NaN, 1]])
    assert.throws(() => selection(a, b, 3));
});
test('pixel mapping uses current viewport and exact end boundary', () => {
  assert.equal(sampleAt(0, 800, [100, 900]), 100);
  assert.equal(sampleAt(400, 800, [100, 900]), 500);
  assert.equal(sampleAt(800, 800, [100, 900]), 900);
  assert.equal(sampleAt(-10, 800, [100, 900]), 100);
});
test('PCM24 sign extension, PCM32 and float32 preserve amplitudes', () => {
  assert.deepEqual(Array.from(parseWav(wav(1, 24, [-8388608, 4194304])).channels[0]), [-1]);
  assert.equal(parseWav(wav(1, 32, [-2147483648, 1073741824])).channels[1][0], 0.5);
  assert.equal(parseWav(wav(3, 32, [-0.5, 0.25])).channels[0][0], -0.5);
});
test('malformed or compressed WAVs fail instead of invented empty tracks', () => {
  assert.throws(() => parseWav(new ArrayBuffer(10)));
  assert.throws(() => parseWav(wav().slice(0, 50)));
  assert.throws(() => parseWav(wav(6)));
  const bad = wav(); new DataView(bad).setUint16(32, 1, true);
  assert.throws(() => parseWav(bad));
});
test('nonfinite float samples are rejected with no silent substitution', () => {
  assert.throws(() => parseWav(wav(3, 32, [NaN, 0])));
  assert.throws(() => parseWav(wav(3, 32, [Infinity, 0])));
});
test('partial final audio frame is rejected', () => {
  assert.throws(() => parseWav(wav(1, 16, [0, 1, 2])));
});
test('odd-sized unknown RIFF chunk is skipped with padding', () => {
  const original = new Uint8Array(wav());
  const bytes = new Uint8Array(original.length + 10);
  bytes.set(original.subarray(0, 36));
  bytes.set([74, 85, 78, 75, 1, 0, 0, 0, 42, 0], 36);
  bytes.set(original.subarray(36), 46);
  new DataView(bytes.buffer).setUint32(4, bytes.length - 8, true);
  assert.equal(parseWav(bytes.buffer).sampleCount, 3);
});
