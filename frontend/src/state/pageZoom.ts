import {ref} from 'vue';import {projects} from '../platform/browser.ts';
export const pageScale=ref(100);let installed=false;
const guarded=new WeakSet<Document>();
function guardWheel(doc:Document){
 if(guarded.has(doc))return;guarded.add(doc);
 doc.defaultView?.addEventListener('wheel',event=>{if(event.ctrlKey)event.preventDefault();},{passive:false,capture:true});
 const attach=(frame:HTMLIFrameElement)=>{try{if(frame.contentDocument)guardWheel(frame.contentDocument);}catch{/* Only app-owned same-origin frames are accessible. */}};
 doc.addEventListener('load',event=>{const target=event.target as HTMLElement|null;if(target?.tagName==='IFRAME')attach(target as HTMLIFrameElement);},true);
 doc.querySelectorAll('iframe').forEach(attach);
}
export function setPageScale(percent:number){
 if(!Number.isFinite(percent))return;
 pageScale.value=Math.max(70,Math.min(150,Math.round(percent)));
 document.documentElement.style.zoom=String(pageScale.value/100);
 document.documentElement.style.setProperty('--page-scale',String(pageScale.value/100));
 projects.write('pageScale',pageScale.value);
}
export function installPageZoom(){
 if(installed)return;installed=true;
 const saved=projects.read<unknown>('pageScale',100);setPageScale(typeof saved==='number'?saved:100);
 // Cancel only the browser's page-zoom default. Plot handlers still receive it.
 guardWheel(document);
}
