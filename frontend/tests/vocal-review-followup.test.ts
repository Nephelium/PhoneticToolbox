import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {editStopTiming} from '../public/vocal-tract/stop-editor.js';

// Exercise the shipped event handlers with only the DOM/host ports replaced.
const keyframes=readFileSync(new URL('../public/vocal-tract/keyframes.js',import.meta.url),'utf8');
const exportHandler=keyframes.slice(keyframes.indexOf("  $('exportFrames').onclick="),keyframes.indexOf("  $('importFrames').onclick="));
for(const outcome of ['saved','cancelled','failed'])test(`draft export after failed autosave: ${outcome}`,async()=>{
 const saveQueue=Promise.reject(Error('storage unavailable'));saveQueue.catch(()=>{});
 const button:any={},calls:any[]=[],messages:string[]=[],locks:boolean[]=[];
 const draft={frames:[{frame_id:'draft',f0:240}],features:[{id:'feature'}],pitch_curve:[[0,240],[1,280]],pressure_curve:[[0,800],[1,900]],perturbation:{jitter:20,shimmer:10,seed:7},global_controls:{lip_width:{mode:'fixed',value:1.2}}};
 const context={$:()=>button,busy:false,editing:-1,saveQueue,frames:draft.frames,
  features:{get:()=>draft.features},pitch:{getCurve:()=>draft.pitch_curve,getPressureCurve:()=>draft.pressure_curve},getPerturbation:()=>draft.perturbation,sequenceControls:{get:()=>draft.global_controls},
  lockFiles:()=>locks.push(true),lock:(v:boolean)=>locks.push(v),text:(v:string)=>messages.push(v),
  post:async(op:string,body:any)=>{calls.push({op,body});if(outcome==='failed')throw Error('export device unavailable');return {json:async()=>({saved:outcome==='saved',name:'draft.json'})};}};
 vm.runInNewContext(exportHandler,context);await button.onclick();
 assert.equal(calls.length,1);assert.equal(calls[0].op,'document/save');assert.deepEqual(JSON.parse(JSON.stringify(calls[0].body)),draft);
 assert.deepEqual(locks,[true,false]);assert.equal(context.saveQueue,saveQueue);
 if(outcome==='saved')assert.match(messages[0],/已导出 draft.json/);
 if(outcome==='cancelled')assert.deepEqual(messages,[]);
 if(outcome==='failed')assert.match(messages[0],/导出失败：export device unavailable/);
});

test('side-region buttons update section, profiles, viewer and chart without a geometry request',()=>{
 const app=readFileSync(new URL('../public/vocal-tract/app.js',import.meta.url),'utf8');
 const handler=app.split(/\r?\n/).find(line=>line.startsWith("document.querySelectorAll('[data-side-region]')"));
 const select=app.slice(app.indexOf('function selectSection(index){'),app.indexOf('function selectArea('));
 const buttons=[1,2,3].map(n=>({dataset:{sideRegion:String(n)},classList:{toggle(){}},onclick:()=>{}}));
 const elements:any={sliceRange:{value:3},slicePosition:{textContent:''}};
 const state={section:3,centerline:Array.from({length:12},(_,i)=>[i,0,i]),airway_sections:Array.from({length:12},(_,i)=>({upper:[i+1],lower:[i]})),upper:[4],lower:[3]};
 let draws=0,charts=0,requests=0;
 // Source-only updates create a new snapshot while retaining viewer geometry.
 const viewerState=structuredClone(state);
 const context={$:(id:string)=>elements[id],selectedSide:2,snapshot:state,dirty:false,queuePose:()=>requests++,drawCharts:()=>charts++,document:{querySelectorAll:()=>buttons},viewer:{state:viewerState,draw2D:()=>draws++,handles:()=>[1,2,3].map((n,i)=>({name:'side'+n,p:[7-i,0]}))}};
 vm.runInNewContext(select+'\n'+handler,context);
 for(const [i,b] of buttons.entries()){b.onclick();const section=7-i;assert.equal(state.section,section);assert.equal(viewerState.section,section);assert.equal(+elements.sliceRange.value,section);assert.deepEqual(state.upper,[section+1]);assert.deepEqual(state.lower,[section]);assert.deepEqual(viewerState.upper,state.upper);assert.deepEqual(viewerState.lower,state.lower);assert.equal(elements.slicePosition.textContent,section.toFixed(1)+' cm');}
 assert.equal(draws,3);assert.equal(charts,3);assert.equal(requests,0);
});

test('stop editing supports the full three-second frame without changing signed VOT rules',()=>{
 const stop={closure:1.992,release:.008,aspiration:0,vot:-.03};
 assert.deepEqual(editStopTiming(stop,'closure',2992),{...stop,closure:2.992});
 assert.throws(()=>editStopTiming(stop,'closure',2993),/3 秒/);
 assert.throws(()=>editStopTiming(stop,'closure',3000),/3 秒/);
 const longest=editStopTiming({...stop,release:.001},'closure',2999);
 assert.equal(longest.closure+longest.release,3);assert.equal(longest.aspiration,0);
 assert.equal(editStopTiming(longest,'vot',-2999).vot,-2.999);
 assert.throws(()=>editStopTiming(longest,'vot',-3000),/前浊/);
});
