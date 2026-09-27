// Test-only transport. Actual Python compute; no production registration or mock result.
import {createApp,h} from 'vue';import Page from '../src/modules/pitch-manipulation/PitchManipulationPage.vue';import '../src/design/tokens.css';
import {selectFontOwner} from '../src/state/fonts.ts';
import type {M08Port,Result} from '../src/modules/pitch-manipulation/port.ts';
import type {ResearchContext} from '../src/platform/research.ts';
async function rpc(op:string,args={}){const response=await fetch('/__m08_test',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({op,...args})});const data=await response.json();if(data.error)throw Error(data.error);return data.value;}
function bytes(text:string){return Uint8Array.from(atob(text),c=>c.charCodeAt(0)).buffer;}
const port:M08Port={async preview(f){const data=await rpc('preview',{id:f.id});return {...data,wav:bytes(data.wav)};},submit:(f,config,key)=>rpc('submit',{id:f.id,config,key}),jobs:()=>rpc('jobs'),cancel:id=>rpc('cancel',{id}),audio:async r=>bytes(await rpc('audio',{id:r.id})),save:r=>rpc('save',{id:r.id}),history:f=>rpc('history',{id:f.id}),remove:ids=>rpc('remove',{ids}),rename:changes=>rpc('rename',{changes}),async download(r:Result){const buffer=bytes(await rpc('audio',{id:r.id}));const url=URL.createObjectURL(new Blob([buffer],{type:'audio/wav'}));const a=document.createElement('a');a.href=url;a.download=r.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}};
const context:ResearchContext={key:'m08-test',label:'M08 synthetic verification',files:{kind:'preview',list:()=>rpc('list'),read:async f=>{const p=await port.preview(f,new AbortController().signal);return {buffer:p.wav,sha256:p.sha256};},textgrid:async()=>{throw Error('not used');},dispose(){}}};
createApp({render:()=>h(Page,{context,stateKey:'m08-browser-test',active:true,port:location.search.includes('unwired')?undefined:port})}).mount('#app');
void selectFontOwner();
