import type { components } from '../../contracts/generated/api';

export type Selection = components['schemas']['Selection'];
export type Track = components['schemas']['Track'];
export type Health = components['schemas']['Health'];

/** Display conversion only. The backend validates cross-field scientific constraints. */
export function selectionDuration(selection: Selection): number {
  return (selection.end_sample - selection.start_sample) / selection.sample_rate_hz;
}
