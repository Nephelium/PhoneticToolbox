import {uiError as displayError,uiMessage as displayMessage} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
export interface ToneSettings {mode:'preserve'|'values'|'classes';classOutput:'original'|'values';mapping:Record<string,string>}
export const defaultToneSettings=():ToneSettings=>({mode:'preserve',classOutput:'original',mapping:{}});
const digits='0123456789',superDigits='⁰¹²³⁴⁵⁶⁷⁸⁹',fullDigits='０１２３４５６７８９';
const normalizeDigits=(value:string)=>[...value].map(c=>digits.includes(c)?c:String(Math.max(superDigits.indexOf(c),fullDigits.indexOf(c)))).join('');
/** Only the explicit numeric suffix is interpreted; existing IPA tone letters stay intact. */
export function toneSuffix(value:string){
  const match=/^(.*?)([0-9⁰¹²³⁴⁵⁶⁷⁸⁹０-９]+[aAbB]?)$/u.exec(value.trim());
  return match?{base:match[1]!.trimEnd(),original:match[2]!,key:normalizeDigits(match[2]!.replace(/[aAbB]$/,'') )+(match[2]!.match(/[aAbB]$/)?.[0].toLowerCase()??'')}:undefined;
}
export function toneSymbols(value:string){
  const text=value.trim();if(/^[˩˨˧˦˥]{1,8}$/u.test(text))return text;
  if(!/^[1-5¹²³⁴⁵１-５]{1,8}$/u.test(text))throw displayError(displayMessage('modules.mandarin-ipa.text.f4a5f9ad4b7e'));
  return [...normalizeDigits(text)].map(c=>'˩˨˧˦˥'[Number(c)-1]).join('');
}
export function toneClasses(rows:readonly (readonly string[])[]){return [...new Set(rows.flatMap(row=>{const tone=toneSuffix(row[1]??'');return tone?[tone.key]:[];}))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));}
export function validateToneSettings(value:unknown):ToneSettings{
  const settings=value as ToneSettings;
  if(!settings||!['preserve','values','classes'].includes(settings.mode)||!['original','values'].includes(settings.classOutput)||!settings.mapping||typeof settings.mapping!=='object'||Array.isArray(settings.mapping))throw displayError(displayMessage('modules.mandarin-ipa.text.3f0d2f80fe13'));
  const mapping=Object.entries(settings.mapping);if(mapping.length>128||mapping.some(([key,target])=>!/^\d+[ab]?$/.test(key)||key.length>16||typeof target!=='string'||target.length>32))throw displayError(displayMessage('modules.mandarin-ipa.text.305a90f762dc'));
  return {mode:settings.mode,classOutput:settings.classOutput,mapping:Object.fromEntries(mapping)};
}
export function transcribeTone(value:string,settings?:ToneSettings,showTone=true){
  if(!settings||settings.mode==='preserve')return showTone?value:value.replace(/[˥˦˧˨˩]+$/u,'');
  const tone=toneSuffix(value);
  if(!tone)return showTone?value:value.replace(/[˥˦˧˨˩]+$/u,'');
  let ending=tone.original;
  if(settings.mode==='values')ending=toneSymbols(tone.original);
  else if(settings.classOutput==='values'){
    const target=Object.hasOwn(settings.mapping,tone.key)?settings.mapping[tone.key]:'';
    if(!target?.trim())throw displayError(displayMessage('modules.mandarin-ipa.text.c1b1e8d4b759',{v0:tone.key}));
    ending=toneSymbols(target);
  }
  return tone.base+(showTone?ending:'');
}
export function tonePreview(rows:readonly (readonly string[])[],settings:ToneSettings){
  if(settings.mode==='classes'&&settings.classOutput==='values'&&toneClasses(rows).length>128)throw displayError(displayMessage('modules.mandarin-ipa.text.1f77bb400af9'));
  return rows.map((row,i)=>{try{return [row[0]!,transcribeTone(row[1]!,settings)] as [string,string];}catch(cause){throw displayError(displayMessage('modules.mandarin-ipa.text.035c4d5b6de6',{v0:i+1,v1:(cause as Error).message}));}});
}
