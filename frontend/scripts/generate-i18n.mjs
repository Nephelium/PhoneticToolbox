import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),base=path.join(root,'src/i18n');
const args=process.argv.slice(2),check=args.includes('--check'),reviewIndex=args.indexOf('--review');
const languages=JSON.parse(fs.readFileSync(path.join(base,'languages.json'),'utf8'));
if(!Array.isArray(languages)||languages[0]?.code!=='zh-CN'||new Set(languages.map(row=>row.code)).size!==languages.length)throw Error('Invalid language registry');
for(const row of languages)if(!/^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$/.test(row.code)||!row.nativeName||!['ltr','rtl'].includes(row.direction))throw Error('Invalid locale metadata');
const files=fs.readdirSync(path.join(base,'locales/zh-CN')).filter(name=>name.endsWith('.json')).sort();
// The plain-JavaScript M10 frame shares canonical catalogs and review checks,
// while loading its own small resources independently of Vue language chunks.
const frameFiles=files.filter(name=>name==='vocal-tract-frame.json');
const vueFiles=files.filter(name=>!frameFiles.includes(name));
const nativeBase=path.resolve(root,'../desktop/src/ptb_desktop/locales'),nativeRows={};
const raw={},errors=[],sourceKeys=new Set(),hash=value=>crypto.createHash('sha256').update(value).digest('hex');
const parameters=value=>[...new Set([...value.matchAll(/\{([A-Za-z_][A-Za-z0-9_]*|[0-9]+)\}/g)].map(match=>match[1]))].sort().join(',');
for(const language of languages){
 const target=path.join(base,'locales',language.code);raw[language.code]={};
 const targetFiles=fs.existsSync(target)?fs.readdirSync(target).filter(name=>name.endsWith('.json')).sort():[];
 if(JSON.stringify(files)!==JSON.stringify(targetFiles))errors.push(language.code+': namespace files differ');
 for(const file of files){
  const full=path.join(target,file);if(!fs.existsSync(full))continue;
  const rows=JSON.parse(fs.readFileSync(full,'utf8'));
  if(!rows||typeof rows!=='object'||Array.isArray(rows))throw Error('Invalid catalogue '+full);
  for(const [key,value] of Object.entries(rows)){
   if(!/^[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*$/.test(key)||typeof value!=='string'||!value.trim())errors.push(language.code+': invalid/empty '+key);
   if(Object.hasOwn(raw[language.code],key))errors.push(language.code+': duplicate '+key);
   raw[language.code][key]=value;if(language.code==='zh-CN')sourceKeys.add(key);
  }
 }
}
// Native Qt and bootstrap messages have one canonical source. The frontend
// consumes the same keys at error boundaries, with no second translation copy.
for(const {code}of languages){
 const native=JSON.parse(fs.readFileSync(path.join(nativeBase,code+'.json'),'utf8'));
 nativeRows[code]=Object.fromEntries(Object.entries(native).map(([key,value])=>['native.'+key,value]));
 for(const [key,value]of Object.entries(nativeRows[code])){
  if(typeof value!=='string'||!value.trim()||Object.hasOwn(raw[code],key))errors.push(code+': invalid native message '+key);
  raw[code][key]=value;if(code==='zh-CN')sourceKeys.add(key);
 }
}
for(const language of languages.slice(1)){
 for(const key of sourceKeys)if(!Object.hasOwn(raw[language.code],key))errors.push(language.code+': missing '+key);else if(parameters(raw[language.code][key])!==parameters(raw['zh-CN'][key]))errors.push(language.code+': named parameters differ '+key);
 for(const key of Object.keys(raw[language.code]))if(!sourceKeys.has(key))errors.push(language.code+': extra '+key);
}
// Registry-owned product notes must invalidate their translations when their
// actual source changes. Bibliography, author names and license texts stay in
// the canonical registry; only these explanatory display fields are projected.
const sourceNotes=new Set();
for(const source of JSON.parse(fs.readFileSync(path.join(root,'src/generated/sources.json'),'utf8'))){
 for(const field of ['reference_note','license','attribution','adaptation_note','adaptation_caution']){
  const value=source[field];if(typeof value!=='string'||!/[\u4e00-\u9fff]/.test(value))continue;
  const key='sources.metadata.'+source.id+'.'+field;sourceNotes.add(key);
  if(raw['zh-CN'][key]!==value)errors.push('Source explanatory note changed or is missing: '+key);
 }
}
for(const key of sourceKeys)if(key.startsWith('sources.metadata.')&&!sourceNotes.has(key))errors.push('Obsolete source explanatory note: '+key);
if(errors.length)throw Error(errors.slice(0,80).join('\n'));
// A review records both source and target identities. A changed Chinese sentence
// cannot silently keep an old translation merely because its key still exists.
const reviewsPath=path.join(base,'reviews.json');let reviews=fs.existsSync(reviewsPath)?JSON.parse(fs.readFileSync(reviewsPath,'utf8')):{};
if(reviewIndex>=0){
 const requested=args[reviewIndex+1];if(!requested)throw Error('--review needs a namespace, or all after an actual review');
 const selected=requested==='all'?[...files,'@native']:requested==='native'?['@native']:files.filter(name=>name===requested+'.json');if(!selected.length)throw Error('Unknown review namespace');
 for(const file of selected){const rows=file==='@native'?nativeRows['zh-CN']:JSON.parse(fs.readFileSync(path.join(base,'locales/zh-CN',file),'utf8'));for(const key of Object.keys(rows))reviews[key]=Object.fromEntries(languages.map(({code})=>[code,hash(raw[code][key])]));}
 fs.writeFileSync(reviewsPath,JSON.stringify(reviews,null,2)+'\n');
}
const reviewErrors=[];
for(const key of sourceKeys)for(const {code}of languages)if(reviews[key]?.[code]!==hash(raw[code][key]))reviewErrors.push(code+': unreviewed/changed '+key);
if(check&&reviewErrors.length)throw Error(reviewErrors.slice(0,80).join('\n'));
const generated=new Map();
// Workers send structured descriptors to the main UI. Their compact Chinese
// fallback keeps direct parser diagnostics readable without loading Vue/i18n.
const workerKeys=new Set();
for(const name of ['platform/wav.ts','platform/wav.worker.ts','platform/vocal.worker.ts']){
 const text=fs.readFileSync(path.join(root,'src',name),'utf8');
 for(const match of text.matchAll(/runtimeError\(\s*['"]([^'"]+)['"]/g))workerKeys.add(match[1]);
}
const workerRows=Object.fromEntries([...workerKeys].sort().map(key=>{
 if(!Object.hasOwn(raw['zh-CN'],key))throw Error('Unknown worker message '+key);
 return [key,raw['zh-CN'][key]];
}));
generated.set('generated/worker-fallback.ts','// Generated by scripts/generate-i18n.mjs; do not edit.\nexport const workerFallbackMessages:Record<string,string>='+JSON.stringify(workerRows,null,2)+';\n');
for(const {code}of languages){
 const imports=vueFiles.map((file,index)=>`import c${index} from '../locales/${code}/${file}' with {type:'json'};`).join('\n')+`\nimport native from './native.${code}.ts';`;
 generated.set('generated/native.'+code+'.ts','// Generated from desktop/src/ptb_desktop/locales; do not edit.\nexport default '+JSON.stringify(nativeRows[code],null,2)+' as Record<string,string>;\n');
 generated.set('generated/'+code+'.ts','// Generated by scripts/generate-i18n.mjs; do not edit.\n'+imports+'\nexport default Object.assign({},'+[...vueFiles.map((_,index)=>'c'+index),'native'].join(',')+') as Record<string,string>;\n');
 if(frameFiles.length){const rows=Object.assign({},nativeRows[code],Object.fromEntries(Object.entries(raw[code]).filter(([key])=>key.startsWith('platform.'))),...frameFiles.map(file=>JSON.parse(fs.readFileSync(path.join(base,'locales',code,file),'utf8'))));generated.set(path.relative(base,path.join(root,'public/vocal-tract/locales',code+'.mjs')),'// Generated by scripts/generate-i18n.mjs; do not edit.\nexport default '+JSON.stringify(rows,null,2)+';\n');}
}
if(frameFiles.length)generated.set(path.relative(base,path.join(root,'public/vocal-tract/locales/loaders.mjs')),'// Generated by scripts/generate-i18n.mjs; do not edit.\nexport const localeLoaders={\n'+languages.slice(1).map(({code})=>JSON.stringify(code)+`:()=>import('./${code}.mjs').then(module=>module.default)`).join(',\n')+'\n};\n');
generated.set('generated/keys.ts','// Generated by scripts/generate-i18n.mjs; do not edit.\nexport type AppLocale = '+languages.map(({code})=>JSON.stringify(code)).join(' | ')+';\nexport type MessageKey = '+[...sourceKeys].sort().map(JSON.stringify).join(' | ')+';\n');
generated.set('generated/loaders.ts',`// Generated by scripts/generate-i18n.mjs; do not edit.\nimport zh from './zh-CN.ts';\nexport const chineseMessages=zh;\nexport const catalogueLoaders={\n${languages.slice(1).map(({code})=>JSON.stringify(code)+`:()=>import('./${code}.ts').then(module=>module.default)`).join(',\n')}\n};\n`);
let changed=[];for(const [name,text]of generated){const full=path.join(base,name);if(!fs.existsSync(full)||fs.readFileSync(full,'utf8')!==text){changed.push(name);if(!check){fs.mkdirSync(path.dirname(full),{recursive:true});fs.writeFileSync(full,text);}}}
if(check&&changed.length)throw Error('Generated catalogue index is stale: '+changed.join(', '));
console.log(JSON.stringify({languages:languages.map(row=>row.code),namespaces:files.length+1,messages:sourceKeys.size,changed,pendingReviews:reviewErrors.length}));
