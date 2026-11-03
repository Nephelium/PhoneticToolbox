import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {msg,tr,join,bindText,appendText,bindAttribute,onLocaleChange,setFrameLocale,uiError,receiveError} from '../public/vocal-tract/i18n.mjs';
import {frameIntervals} from '../public/vocal-tract/keyframes.js';
import {localizedPresetChart,presetChart} from '../src/modules/vocal-tract/ipa-chart.ts';
import {setLocale} from '../src/i18n/index.ts';
import {messageSnapshot,uiMessage,joinMessages} from '../src/modules/module-localization.ts';

class TextNode {
 nodeType=3;parentNode:any=null;ownerDocument:any;data:string;
 constructor(text='',owner?:unknown){this.data=text;this.ownerDocument=owner;}
 get textContent(){return this.data;}set textContent(value:string){this.data=value;}
 get nodeValue(){return this.data;}set nodeValue(value:string){this.data=value;}
 get isConnected():boolean{return !!this.parentNode?.isConnected;}
}
class Element {
 nodeType=1;parentNode:any=null;ownerDocument:any;children:any[]=[];attributes=new Map<string,string>();root=false;
 constructor(owner?:unknown){this.ownerDocument=owner;}
 get firstChild(){return this.children[0]??null;}
 get textContent():string{return this.children.map(n=>n.textContent).join('');}
 set textContent(value:string){for(const child of this.children)child.parentNode=null;this.children=[];if(value)this.append(new TextNode(value,this.ownerDocument));}
 get isConnected():boolean{return this.root||!!this.parentNode?.isConnected;}
 append(node:any){node.parentNode=this;this.children.push(node);}
 setAttribute(name:string,value:string){this.attributes.set(name,value);}
}
const document={documentElement:new Element(),createTextNode:(value:string)=>new TextNode(value,document),dispatchEvent:()=>true};
document.documentElement.root=true;document.documentElement.ownerDocument=document;
const attach=()=>{const node=new Element(document);document.documentElement.append(node);return node;};
const keys={saved:'modules.vocal-tract-frame.text.current_keyframe_saved_148',failed:'modules.vocal-tract-frame.text.save_failed_125',lips:'modules.vocal-tract-frame.parameters.LP'};

test('M10 bridge snapshots retain nested descriptors and literal diagnostics across delayed locale changes',async()=>{
 const previous=(globalThis as any).document;(globalThis as any).document=document;
 try{
  await setLocale('zh-CN');
  const wire=JSON.parse(JSON.stringify(messageSnapshot(joinMessages(uiMessage('platform.invalid_saved_data'),' user 原文 007',uiMessage('platform.original_data_is_preserved_export_a_recovery_copy')))));
  assert.equal(wire.type,'joined');assert.equal(wire.parts[0].type,'message');
  const error=receiveError(wire,'raw fallback'),node=attach();bindText(node,()=>error.message);
  await setFrameLocale('en');assert.match(node.textContent,/Invalid saved data\. user 原文 007.*Original data is preserved/);
  await setFrameLocale('ja');assert.match(node.textContent,/user 原文 007/);assert.doesNotMatch(node.textContent,/Original data/);
  const nested=messageSnapshot(uiMessage('modules.vocal-tract-frame.pack.summary',{speaker:uiMessage('modules.vocal-tract-frame.text.female_209'),count:1}));
  const packageError=receiveError(JSON.parse(JSON.stringify(nested)),'unused');
  await setFrameLocale('en');assert.match(packageError.message,/Female vocal-tract model · Symbols: 1/);
  await setFrameLocale('ja');assert.match(packageError.message,/女性声道モデル/);
  assert.equal(receiveError({type:'message',key:'untrusted.key'},'外部ログ 123').message,'外部ログ 123');
  const cycle:any={key:'platform.invalid_saved_data',values:{}};cycle.values.child=cycle;
  assert.throws(()=>messageSnapshot(cycle),/Cyclic/);
  let deep:any='literal';for(let i=0;i<20;i++)deep={type:'joined',parts:[deep]};
  assert.equal(receiveError(deep,'bounded diagnostic').message,'bounded diagnostic');
 }finally{await setFrameLocale('zh-CN');await setLocale('zh-CN');(globalThis as any).document=previous;}
});

test('M10 cached labels/errors change language without replacing controls, scientific values, or sequence names',async()=>{
 const previous=(globalThis as any).document;(globalThis as any).document=document;
 const frames=[{name:'作者的姿势',duration:.2,f0:120,params:[1,2],silent:false},{duration:.3,silent:true,f0:0}],before=JSON.stringify(frames),intervals=JSON.stringify(frameIntervals(frames));
 let redraws=0;const off=onLocaleChange(()=>redraws++);
 try{
  await setFrameLocale('zh-CN');
  const label=attach(),code=new Element(document),input={value:'0.',focused:true};code.textContent='LP';
  bindText(label,()=>msg(keys.lips));label.append(code);const text=label.firstChild;
  const status=attach(),error=uiError(join(msg(keys.failed),'raw diagnostic 007'));
  bindText(status,()=>error.message);bindAttribute(label,'aria-label',()=>msg(keys.lips));
  assert.match(status.textContent,/保存失败/);
  const results=await Promise.all([setFrameLocale('ja'),setFrameLocale('en')]);
  assert.equal(document.documentElement.lang,'en');assert.equal(results.at(-1),true);
  assert.equal(label.firstChild,text);assert.equal(label.children[1],code);assert.equal(code.textContent,'LP');
  assert.equal(label.textContent,'Lip protrusionLP');assert.match(status.textContent,/Save failed: raw diagnostic 007/);
  assert.equal(label.attributes.get('aria-label'),'Lip protrusion');
  const unchanged=redraws;await setFrameLocale('en');assert.equal(redraws,unchanged);
  await setFrameLocale('ja');assert.equal(label.textContent,'唇の突出LP');assert.match(status.textContent,/保存失敗：raw diagnostic 007/);
  assert.deepEqual(input,{value:'0.',focused:true});assert.equal(JSON.stringify(frames),before);assert.equal(JSON.stringify(frameIntervals(frames)),intervals);
 }finally{off();await setFrameLocale('zh-CN');(globalThis as any).document=previous;}
});

test('M10 append bindings retain structured notices and unknown host diagnostics literally',async()=>{
 const previous=(globalThis as any).document;(globalThis as any).document=document;
 try{
  const node=attach();bindText(node,msg(keys.saved));appendText(node,' / E_EXTERNAL 原始日志');
  const error=receiveError({key:keys.saved},'unused');const unknown=receiveError({key:'unknown.key'},'E_EXTERNAL 原始日志');
  await setFrameLocale('en');assert.equal(node.textContent,'Current keyframe saved. / E_EXTERNAL 原始日志');assert.equal(error.message,'Current keyframe saved.');assert.equal(unknown.message,'E_EXTERNAL 原始日志');
  await setFrameLocale('ja');assert.equal(error.message,'現在のキーフレームを保存しました。');assert.equal(unknown.message,'E_EXTERNAL 原始日志');
  assert.equal(tr('unknown.scientific.{V}'),'unknown.scientific.{V}');
 }finally{await setFrameLocale('zh-CN');(globalThis as any).document=previous;}
});

test('M10 frame messages remain separate from Vue chunks and use explicit binding keys',()=>{
 const index=readFileSync(new URL('../src/i18n/generated/en.ts',import.meta.url),'utf8');assert.doesNotMatch(index,/vocal-tract-frame/);
 const html=readFileSync(new URL('../public/vocal-tract/index.html',import.meta.url),'utf8');
 assert.match(html,/data-m10-i18n-text="modules\.vocal-tract-frame\./);assert.doesNotMatch(html,/<title><!--m10-i18n/);
 assert.match(html,/<!--m10-i18n:modules\.vocal-tract-frame\./);
 const runtime=readFileSync(new URL('../public/vocal-tract/i18n.mjs',import.meta.url),'utf8');assert.doesNotMatch(runtime,/location\.(?:replace|reload)|post\(|request\(/);
});

test('M10 localized picker projections cache per locale while retaining every scientific identity',async()=>{
 const source=JSON.stringify(presetChart);
 try{
  await setLocale('en');const en=localizedPresetChart();assert.equal(localizedPresetChart(),en);
  await setLocale('ja');const ja=localizedPresetChart();assert.equal(localizedPresetChart(),ja);assert.notEqual(en,ja);
  assert.deepEqual(en.entries.map(({id,display,insertText})=>({id,display,insertText})),presetChart.entries.map(({id,display,insertText})=>({id,display,insertText})));
  assert.deepEqual(en.consonants.rows.map(row=>row.cells),presetChart.consonants.rows.map(row=>row.cells));assert.deepEqual(ja.vowels,presetChart.vowels);
  assert.notEqual(en.entries[0].nameZh,ja.entries[0].nameZh);
  await setLocale('en');assert.equal(localizedPresetChart(),en);assert.equal(JSON.stringify(presetChart),source);
 }finally{await setLocale('zh-CN');}
});
