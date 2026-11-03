import {catalog,localizedSymbol} from '../ipa-plus/catalog.ts';
import {localizedChartSections} from '../ipa-plus/chart-localization.ts';
import {locale} from '../../i18n/index.ts';
import {buildPresetChart} from './preset-chart.ts';

// The scientific picker source is shared with M17; presentation is cached once
// per language, with all symbol IDs and insertion strings preserved.
export const presetChart=buildPresetChart(catalog);
const cache=new Map<string,ReturnType<typeof buildPresetChart>>();
export function localizedPresetChart(){
  const language=locale.value,saved=cache.get(language);if(saved)return saved;
  const chart=buildPresetChart({...catalog,entries:catalog.entries.map(localizedSymbol),charts:{...catalog.charts,ipa:localizedChartSections('ipa',catalog.charts.ipa)}});
  cache.set(language,chart);return chart;
}
