import {catalog} from '../ipa-plus/catalog.ts';
import {buildPresetChart} from './preset-chart.ts';

// Use the existing M17 source and Chinese names for M10's compact picker.
export const presetChart=buildPresetChart(catalog);
