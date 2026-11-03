import {locale,tr} from '../../i18n/index.ts';
import type {ChartSection,ChartSystem} from './types.ts';

// Keyboard and full charts keep separate source projections. Only display fields
// are copied, once for each language; symbol IDs and chart geometry are retained.
const cache=new WeakMap<ChartSection[],Map<string,ChartSection[]>>();
let builds=0;
export function localizedChartSections(system:ChartSystem,sections:ChartSection[]):ChartSection[]{
 const language=locale.value;
 let languages=cache.get(sections);if(!languages){languages=new Map();cache.set(sections,languages);}
 const saved=languages.get(language);if(saved)return saved;
 const result=sections.map(section=>{
  const prefix=`modules.ipa-plus.chart.${system}.${section.id}`;
  return {...section,title:tr(prefix+'.title'),subtitle:section.subtitle?tr(prefix+'.subtitle'):section.subtitle,
   columns:section.columns?.map((_,i)=>tr(`${prefix}.columns.${i}`)),
   rows:section.rows?.map((row,i)=>({...row,label:tr(`${prefix}.rows.${i}.label`)})),
   groups:section.groups?.map((group,i)=>({...group,label:tr(`${prefix}.groups.${i}.label`),hint:group.hint?tr(`${prefix}.groups.${i}.hint`):group.hint})),
  };
 });
 builds++;languages.set(language,result);return result;
}
export function chartLocalizationBuilds(){return builds;}
