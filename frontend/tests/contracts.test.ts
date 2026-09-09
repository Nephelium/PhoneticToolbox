import test from 'node:test';
import assert from 'node:assert/strict';
import { selectionDuration } from '../src/contracts.ts';
import type { Track } from '../src/contracts.ts';

test('sample-frame duration preserves half-open EOF and empty selection', () => {
  assert.equal(selectionDuration({ start_sample: 123, end_sample: 22173, sample_rate_hz: 44100 }), 0.5);
  assert.equal(selectionDuration({ start_sample: 88199, end_sample: 88200, sample_rate_hz: 44100 }), 1 / 44100);
  assert.equal(selectionDuration({ start_sample: 88200, end_sample: 88200, sample_rate_hz: 44100 }), 0);
});

test('JSON transport preserves separate unvoiced and failed frames', () => {
  const track: Track = { parameter_key: 'pF0', backend: 'praat', unit: 'Hz',
    times_s: [0, 0.01, 0.02], values: [120, null, null], validity: ['valid', 'unvoiced', 'failed'],
    reason: [null, 'unvoiced', 'backend_failure'], analysis_config_hash: 'a'.repeat(64), source_ids: ['SRC-PRAAT'] };
  assert.deepEqual(JSON.parse(JSON.stringify(track)), track);
});
