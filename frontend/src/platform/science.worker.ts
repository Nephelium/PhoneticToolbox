/// <reference lib="webworker" />
// Fixed operations only. Research bytes never enter generated Python source.
import runtimeLock from '../../../resources/manifests/browser-science-runtime.json';
declare const loadPyodide:any;
declare const createPtbPraat:any;
const base=new URL('../browser-science/',self.location.href);
function pinned(name:string,sha:string){const url=new URL(name,base);url.searchParams.set('sha256',sha);return url.href;}
async function checked(name:string,sha:string){
 const response=await fetch(pinned(name,sha));if(!response.ok)throw Error('Browser core unavailable');const bytes=await response.arrayBuffer();
 const actual=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),n=>n.toString(16).padStart(2,'0')).join('');
 if(actual!==sha)throw Error('Browser core identity mismatch');return bytes;
}
let python:any,ready:Promise<void>|undefined,praatReady:Promise<void>|undefined;
let fontsReady:Promise<void>|undefined;
async function renderFonts(){
 if(!fontsReady)fontsReady=(async()=>{
  const data=await checked('fonts/NotoSansSC-VF.ttf',runtimeLock.render_fonts['NotoSansSC-VF.ttf'].sha256);
  python.FS.writeFile('/ptb/ptb_worker/assets/NotoSansSC-VF.ttf',new Uint8Array(data));
  python.runPython('from matplotlib import font_manager\nfont_manager.fontManager.addfont("/ptb/ptb_worker/assets/NotoSansSC-VF.ttf")');
 })();await fontsReady;
}
async function initialize(){
 if(!ready)ready=(async()=>{
  importScripts(pinned('runtime/pyodide.js',runtimeLock.files['pyodide.js'].sha256));
  python=await loadPyodide({indexURL:new URL('runtime/',base).href});
  python.unpackArchive(await checked('ptb-browser.zip',runtimeLock.archive_sha256),'zip',{extractDir:'/ptb'});
  python.runPython('import sys;sys.path.insert(0,"/ptb")');
 })();await ready;
}
async function nativePraat(){
 if(!praatReady)praatReady=(async()=>{
  importScripts(pinned('ptb_praat.js',runtimeLock.native_praat['ptb_praat.js'].sha256));
  const wasmBinary=new Uint8Array(await checked('ptb_praat.wasm',runtimeLock.native_praat['ptb_praat.wasm'].sha256));
  const module=await createPtbPraat({wasmBinary,locateFile:(name:string)=>new URL(name,base).href});
  python.globals.set('_ptb_praat_wasm',module);
  python.runPython('from phonetic_core.browser import praat\npraat.module=_ptb_praat_wasm\nsys.modules["parselmouth"]=praat');
 })();await praatReady;
}
self.onmessage=async event=>{
 const {id,operation,raw,args}=event.data;
 let initialized=false;
 try{
  await initialize();
  if(operation==='parameters')await python.loadPackage(['sqlite3']);
  else if(operation==='phonology'){
   await python.loadPackage(['lxml','pydantic']);
   if(args.config?.computation_revision!=='m14/2'&&args.config?.action!=='inspect')await python.loadPackage(['pandas']);
  }
  else{
   await python.loadPackage(['numpy','scipy','soundfile']);
   // A type-only bridge is enough for pure NumPy/SciPy operations. Load the
   // 128 MB original Praat module only when an operation actually calls it.
   python.runPython('from phonetic_core.browser import praat\nsys.modules["parselmouth"]=praat');
   if(operation==='spectrogram'||operation==='recording_spectrum'||operation==='egg_update'&&args.config?.keep_praat_f0)await nativePraat();
   if(operation.startsWith('egg_'))await python.loadPackage(['pydantic','pillow','matplotlib']);
   if(operation==='prediction'||operation==='reconstruction')await python.loadPackage(['pydantic','pillow','opencv-python']);
   if(operation==='recording_compute')await python.loadPackage(['pydantic']);
   if(operation==='lpc'||operation==='lpc_fonts'){await python.loadPackage(['pydantic','matplotlib']);await renderFonts();}
  }
  initialized=true;python.globals.set('_ptb_operation',operation);python.globals.set('_ptb_raw',new Uint8Array(raw));python.globals.set('_ptb_args',JSON.stringify(args));
  try{
   // Reuse the prediction guard before automatic GL, too: small CNN kernel
   // differences must not choose opposite sides of an integer window boundary.
   if(operation==='reconstruction'&&args.auto_parameters&&args.mode!=='audio_draw')python.runPython('from ptb_worker.browser_compute import dispatch\ndispatch("prediction",_ptb_raw,_ptb_args)');
   const result=python.runPython('from ptb_worker.browser_compute import dispatch\ndispatch(_ptb_operation,_ptb_raw,_ptb_args)');
   self.postMessage({id,result:JSON.parse(result)});
  }finally{python.runPython('del _ptb_operation,_ptb_raw,_ptb_args');}
 }catch(error){self.postMessage({id,error:error instanceof Error?error.message:String(error),fatal:!initialized});}
};
