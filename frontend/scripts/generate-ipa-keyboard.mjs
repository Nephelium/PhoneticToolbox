// Compact projection of the authoritative Plus catalog, with no explanatory prose.
import fs from 'node:fs';
const source=new URL('../src/modules/ipa-plus/data/catalog.json',import.meta.url);
const target=new URL('../src/modules/ipa-plus/data/keyboard.json',import.meta.url);
const catalog=JSON.parse(fs.readFileSync(source,'utf8'));
const charts=Object.fromEntries(Object.entries(catalog.charts).map(([system,sections])=>[system,sections.map(section=>{
 const {id,title,kind,columns,rows,ids,points}=section;return {id,title,kind,columns,rows,ids,points};
})]));
const used=new Set(Object.values(charts).flatMap(sections=>sections.flatMap(s=>s.ids)));
const entries=Object.fromEntries(catalog.entries.filter(e=>used.has(e.id)).map(e=>{
 const {id,display,insertText,insertionMode,prefix,suffix}=e;return [id,{display,insertText,insertionMode,prefix,suffix}];
}));
const text=JSON.stringify({version:catalog.version,charts,entries})+'\n';
if(process.argv.includes('--check')){if(!fs.existsSync(target)||fs.readFileSync(target,'utf8')!==text)throw Error('IPA keyboard data is stale; run node scripts/generate-ipa-keyboard.mjs');}
else fs.writeFileSync(target,text,'utf8');
