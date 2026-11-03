/// <reference lib="webworker" />
import scienceLock from '../../../resources/manifests/browser-science-runtime.json';
import vocalLock from '../../../resources/manifests/vocal-wasm-runtime.json';
import {runtimeError,errorDisplay} from '../i18n/runtime-errors.ts';
declare const loadPyodide:any;
const science=new URL('../browser-science/',self.location.href),vocal=new URL('../vocal-wasm/',self.location.href);
let python:any,ready:Promise<void>|undefined;
function pinned(base:URL,name:string,sha:string){const url=new URL(name,base);url.searchParams.set('sha256',sha);return url.href;}
async function checked(base:URL,name:string,sha:string){
 const response=await fetch(pinned(base,name,sha));if(!response.ok)throw runtimeError('platform.the_browser_vocal_tract_runtime_is_temporarily_unavailable');
 const raw=await response.arrayBuffer(),actual=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),b=>b.toString(16).padStart(2,'0')).join('');
 if(actual!==sha)throw runtimeError('platform.browser_vocal_tract_runtime_verification_failed');return raw;
}
async function initialize(checkpoint:any){
 if(!ready)ready=(async()=>{
  importScripts(pinned(science,'runtime/pyodide.js',scienceLock.files['pyodide.js'].sha256));
  python=await loadPyodide({indexURL:new URL('runtime/',science).href});
  await python.loadPackage(['numpy','scipy']);
  python.unpackArchive(await checked(vocal,vocalLock.archive.name,vocalLock.archive.sha256),'zip',{extractDir:'/vocal'});
  for(const name of ['libVocalTractLabApi.so','libVocalTractLabAnalysis.so','libgeometry_p2.so']){
   await python._module.loadDynamicLibrary('/vocal/vocal-native/emscripten-wasm32/'+name,{loadAsync:true,global:false,nodelete:true});
  }
  python.globals.set('_checkpoint',JSON.stringify(checkpoint??null));
  python.runPython('import sys,json\nsys.path.insert(0,"/vocal")\nfrom ptb_worker.browser_vocal import BrowserVocalRuntime\n_vocal_host=BrowserVocalRuntime("/vocal/vocal-native",json.loads(_checkpoint))\ndel _checkpoint');
 })();await ready;
}
self.onmessage=async event=>{
 const {id,op,body,checkpoint}=event.data;let initialized=false;
 try{
  await initialize(checkpoint);initialized=true;
  python.globals.set('_op',op);python.globals.set('_body',JSON.stringify(body));
  try{const result=JSON.parse(python.runPython('_vocal_host.dispatch(_op,_body)'));self.postMessage({id,...result});}
  finally{python.runPython('del _op,_body');}
 }catch(error){console.error(error);const display=errorDisplay(error)??runtimeError('platform.browser_vocal_tract_computation_was_interrupted_retry_the_operation').display;self.postMessage({id,error:display.key,display,fatal:!initialized});}
};
