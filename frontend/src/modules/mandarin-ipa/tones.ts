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
  if(!/^[1-5¹²³⁴⁵１-５]{1,8}$/u.test(text))throw Error('调值请填写 1–5 组成的五度调值，例如 55、35、214，或声调符号。');
  return [...normalizeDigits(text)].map(c=>'˩˨˧˦˥'[Number(c)-1]).join('');
}
export function toneClasses(rows:readonly (readonly string[])[]){return [...new Set(rows.flatMap(row=>{const tone=toneSuffix(row[1]??'');return tone?[tone.key]:[];}))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));}
export function validateToneSettings(value:unknown):ToneSettings{
  const settings=value as ToneSettings;
  if(!settings||!['preserve','values','classes'].includes(settings.mode)||!['original','values'].includes(settings.classOutput)||!settings.mapping||typeof settings.mapping!=='object'||Array.isArray(settings.mapping))throw Error('声调设置无法读取，请重新设置。');
  const mapping=Object.entries(settings.mapping);if(mapping.length>128||mapping.some(([key,target])=>!/^\d+[ab]?$/.test(key)||key.length>16||typeof target!=='string'||target.length>32))throw Error('声调映射内容超出支持范围。');
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
    if(!target?.trim())throw Error(`请为调类 ${tone.key} 填写调值。`);
    ending=toneSymbols(target);
  }
  return tone.base+(showTone?ending:'');
}
export function tonePreview(rows:readonly (readonly string[])[],settings:ToneSettings){
  if(settings.mode==='classes'&&settings.classOutput==='values'&&toneClasses(rows).length>128)throw Error('调类超过 128 种，请检查音标列及声调表示方式。');
  return rows.map((row,i)=>{try{return [row[0]!,transcribeTone(row[1]!,settings)] as [string,string];}catch(cause){throw Error(`第 ${i+1} 条记录：${(cause as Error).message}`);}});
}
