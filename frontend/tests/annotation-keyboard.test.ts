import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';import {execFileSync} from 'node:child_process';
import compact from '../src/modules/ipa-plus/data/keyboard.json' with {type:'json'};
import original from '../src/modules/ipa-plus/data/catalog.json' with {type:'json'};
import {insertSymbol} from '../src/modules/ipa-plus/editor.ts';
test('M12 compact keyboard uses the canonical Plus layout and exact Unicode insertion modes without explanatory payload',()=>{
 execFileSync(process.execPath,['scripts/generate-ipa-keyboard.mjs','--check'],{cwd:new URL('..',import.meta.url)});
 for(const [system,sections] of Object.entries(compact.charts))for(const section of sections){const source=original.charts[system as keyof typeof original.charts].find(s=>s.id===section.id)!;assert.deepEqual(section.ids,source.ids);}
 for(const [id,item] of Object.entries(compact.entries)){const source=original.entries.find(e=>e.id===id)!;for(const key of Object.keys(item))assert.deepEqual(item[key as keyof typeof item],source[key as keyof typeof source]);assert(!('descriptionZh' in item));}
 assert(fs.statSync(new URL('../src/modules/ipa-plus/data/keyboard.json',import.meta.url)).size<90000);
});
test('M12 keyboard insertion replaces selection, preserves combining marks and wraps spans with the same Plus rules',()=>{
 const values=Object.values(compact.entries),literal=values.find(e=>e.display==='æ')!;
 assert.deepEqual(insertSymbol({text:'汉xy语',start:1,end:3},literal as any),{text:'汉æ语',start:2,end:2});
 const combining=values.find(e=>e.insertionMode==='combining')!;const inserted=insertSymbol({text:'a',start:1,end:1},combining as any);assert.equal(inserted.text,'a'+combining.insertText);
 const paired=values.find(e=>e.insertionMode==='paired-span')!;assert.equal(insertSymbol({text:'a',start:0,end:1},paired as any).text,(paired as any).prefix+'a'+(paired as any).suffix);
});
