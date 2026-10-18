/* QA only: executes the same source recipes against the actual browser ABI. */
self.onmessage=async()=>{
 try{
  const start=performance.now(),science=new URL('../browser-science/',self.location.href),vocal=new URL('../vocal-wasm/',self.location.href);
  importScripts(new URL('runtime/pyodide.js',science).href);
  const py=await loadPyodide({indexURL:new URL('runtime/',science).href});
  await py.loadPackage(['numpy','scipy']);
  const manifest=await(await fetch(new URL('manifest.json',vocal))).json(),raw=await(await fetch(new URL(manifest.archive.name,vocal))).arrayBuffer();
  const sha=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),b=>b.toString(16).padStart(2,'0')).join('');
  if(sha!==manifest.archive.sha256)throw Error('QA bundle identity mismatch');
  py.unpackArchive(raw,'zip',{extractDir:'/vocal'});
  for(const name of ['libVocalTractLabApi.so','libVocalTractLabAnalysis.so','libgeometry_p2.so'])await py._module.loadDynamicLibrary('/vocal/vocal-native/emscripten-wasm32/'+name,{loadAsync:true,global:false,nodelete:true});
  const code=await(await fetch(new URL('vocal_wasm_cases.py',self.location.href))).text();py.FS.writeFile('/vocal/vocal_wasm_cases.py',code);
  const loaded=performance.now();
  const result=py.runPython('import sys,json\nsys.path.insert(0,"/vocal")\nfrom vocal_wasm_cases import run\njson.dumps(run("/vocal/vocal-native"),ensure_ascii=False,separators=(",",":"))');
  self.postMessage({result,load_ms:loaded-start,compute_ms:performance.now()-loaded,heap_bytes:py._module.HEAPU8.buffer.byteLength,user_agent:self.navigator.userAgent});
 }catch(error){self.postMessage({error:String(error)});}
};
