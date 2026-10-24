// Test-only production build: preserve OL calls/results while measuring projection.
import {build, loadConfigFromFile} from '../../frontend/node_modules/vite/dist/node/index.js';
import {resolve,basename} from 'node:path';
import {writeFile, readFile, readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const root=resolve(import.meta.dirname,'../..'),out=resolve(process.argv[2]);
const rendererOnly=process.env.PTB_GEO_RENDERER_ONLY==='1';
const frontend=resolve('F:/PhoneticToolboxData/geolinguistics-builds','frontend-datav-'+(rendererOnly?'renderer-':'performance-')+basename(out));
const sha=b=>createHash('sha256').update(b).digest('hex');
const files=[...(await readdir(resolve(root,'frontend/src/modules/geolinguistics'))).filter(p=>/\.(ts|vue)$/.test(p)).map(p=>'src/modules/geolinguistics/'+p),
  ...(await readdir(resolve(root,'frontend/src/platform'))).filter(p=>p.startsWith('geolinguistics')&&p.endsWith('.ts')).map(p=>'src/platform/'+p)];
const sourceHashes=()=>Promise.all(files.map(async p=>[p,sha(await readFile(resolve(root,'frontend',p)))]));
const beforeSources=Object.fromEntries(await sourceHashes());
const loaded=await loadConfigFromFile({command:'build',mode:'production'},resolve(root,'frontend/vite.config.ts'));
let transformed=0,observedPage=0;
const instrument={name:'ptb-owned-ol-readfeatures-observer',enforce:'pre',transform(code,id){
  if(rendererOnly&&id.replaceAll('\\','/').endsWith('/src/modules/geolinguistics/GeoMap.vue'))return {code:code.replace('</script>',String.raw`
const __qaRendererKey={};
onMounted(()=>{const entries=((window as any).__ptbRendererEntries??=new Map());entries.set(__qaRendererKey,()=>({
 active:!!map?.getTargetElement()?.isConnected,zoom:map?.getView().getZoom(),level:currentLevel(),
 layers:layers.map((layer,index)=>({index,features:layer.getSource()?.getFeatures().length??0,
  hasAdmin:!!layer.getSource()?.getFeatures().some(f=>!!f.get('level'))})),
 territory:territory.getFeatures().map(f=>{const fn=f.getStyleFunction()??layers[1].getStyleFunction(),s=fn?.(f,map!.getView().getResolution()!);return {
  id:f.getId(),boundaryType:f.get('boundaryType'),role:f.get('renderRole'),styles:Array.isArray(s)?s.length:s?1:0};})
}));});
onBeforeUnmount(()=>{(window as any).__ptbRendererEntries?.delete(__qaRendererKey);});
</script>`),map:null};
  if(id.replaceAll('\\','/').endsWith('/src/modules/geolinguistics/GeolinguisticsPage.vue')){
    observedPage++;
    return {code:code.replace('</script>',String.raw`
const __ptbMapRefSnapshot=()=>({entries:maps.value.map((v,index)=>({index,present:!!v,
  hasExport:typeof v?.exportSvg==='function',hasReload:typeof v?.reload==='function'}))});
(window as any).__ptbMapRefSnapshot=__ptbMapRefSnapshot;
onBeforeUnmount(()=>{if((window as any).__ptbMapRefSnapshot===__ptbMapRefSnapshot)delete (window as any).__ptbMapRefSnapshot;});
</script>`),map:null};
  }
  if(!id.replaceAll('\\','/').endsWith('/ol/format/GeoJSON.js'))return;
  transformed++;
  return {code:code+String.raw`
const __ptbOriginalReadFeatures=GeoJSON.prototype.readFeatures;
const __ptbObjects=new WeakMap(),__ptbFormats=new WeakMap();let __ptbObjectId=0,__ptbFormatId=0;
GeoJSON.prototype.readFeatures=function(source,options){
 const object=source!==null&&typeof source==='object';
 if(object&&!__ptbObjects.has(source))__ptbObjects.set(source,++__ptbObjectId);
 if(!__ptbFormats.has(this))__ptbFormats.set(this,++__ptbFormatId);
 const began=performance.now();let result,error;
 try{return result=__ptbOriginalReadFeatures.call(this,source,options);}
 catch(e){error=String(e);throw e;}
 finally{window.__ptbGeoPerf.reads.push({stage:window.__ptbGeoPerf.stage,start:began,ms:performance.now()-began,
  objectId:object?__ptbObjects.get(source):null,formatId:__ptbFormats.get(this),inputType:typeof source,
  features:source?.features?.length??null,firstId:source?.features?.[0]?.id??null,returned:result?.length??null,
  dataProjection:options?.dataProjection,featureProjection:options?.featureProjection,error});}
};
`,map:null};
}};
await build({...loaded.config,configFile:false,root:resolve(root,'frontend'),cacheDir:resolve(out,'scratch/vite-cache'),
  plugins:[...loaded.config.plugins,instrument],build:{...loaded.config.build,outDir:frontend}});
if(transformed!==1||observedPage!==1)throw Error('Expected one OL/one Page observer, got '+transformed+'/'+observedPage);
const afterSources=Object.fromEntries(await sourceHashes());
if(JSON.stringify(beforeSources)!==JSON.stringify(afterSources))throw Error('Business sources changed during observer build');
await writeFile(resolve(out,'build.json'),JSON.stringify({success:true,frontend,transformed,observedPage,rendererOnly,indexSha256:sha(await readFile(resolve(frontend,'index.html'))),
  sources:afterSources,sourceStableAcrossBuild:true},null,2));
