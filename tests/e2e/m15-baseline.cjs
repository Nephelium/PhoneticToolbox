// Independent execution of unmodified V2 function bodies. No V3 imports.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),source=path.resolve(root,'../PhoneticToolbox_v2/phonetic_toolbox/gui/resources/perception_experiment/perception_experiment.html');
const html=fs.readFileSync(source,'utf8');
function between(a,b){return html.slice(html.indexOf(a),html.indexOf(b,html.indexOf(a)));}
const helpers=between('const shuffleInPlace =','// === 渲染组件');
const body=between('const startActualTrial = useCallback(async () => {','}, [config, currentItem, status, isComplex, files]);').replace('const startActualTrial = useCallback(async () => {','async function run(){')+'}';
async function capture(type,isi=500,fail=false,missing=false){const log=[];const box={config:{isi,useBeep:false},currentItem:type==='audio'?{type,fileId:'X'}:{type,stimuli:{A:{fileId:missing?null:'A'},B:{fileId:'B'},X:{fileId:'X'}}},status:'waiting',isComplex:type!=='audio',files:['A','B','X'].map(id=>({id,url:id})),console:{warn(){},error(){}},Date:{now:()=>12345},setStatus:s=>log.push(['status',s]),setStartTime:t=>log.push(['start',t]),setPlayingStimulus:s=>log.push(['role',s]),setTimeout:(f,ms)=>{log.push(['wait',ms]);f();},playBeep(){}};
const handlers={};box.audioRef={current:{addEventListener:(k,f)=>handlers[k]=f,removeEventListener(){},play(){log.push(['play',this.src]);if(fail)return Promise.reject(Error('failed'));queueMicrotask(()=>handlers.ended());return Promise.resolve();}}};
vm.createContext(box);vm.runInContext(body+';globalThis.execute=run;',box);await box.execute();return log;}
(async()=>{const output={source,sha256:crypto.createHash('sha256').update(html).digest('hex'),cases:{}};
for(const [type,expected] of [['audio',['X']],['ax',['A','X']],['abx',['A','B','X']],['axb',['A','X','B']]]){const log=await capture(type);assert.deepEqual(log.filter(x=>x[0]==='play').map(x=>x[1]),expected);assert.deepEqual(log.slice(-2),[['status','responding'],['start',12345]]);output.cases[type]=log;}
output.cases.zeroISI=await capture('ax',0);assert(output.cases.zeroISI.some(x=>x[0]==='wait'&&x[1]===500));
output.cases.failed=await capture('ax',500,true);output.cases.missing=await capture('ax',500,false,true);
const box={Math:Object.assign(Object.create(Math),{random:()=>0})};vm.createContext(box);vm.runInContext(helpers+';globalThis.result={range:getKeyRangeForTrial({keyRanges:[{start:1,end:3,keys:[]},{start:2,end:4}]},1),keys:getEffectiveKeys({validKeys:[{key:"f"}],keyRanges:[{start:1,end:3,keys:[]}]},1),shuffle:(()=>{const a=[0,1,2,3,4];shuffleInPlace(a,1,3);return a})()};',box);output.helpers=JSON.parse(JSON.stringify(box.result));assert.deepEqual(output.helpers.shuffle,[0,2,3,1,4]);assert.equal(output.helpers.range.start,1);
const out=path.join(root,'output/validation/m15-baseline');fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,'baseline.json'),JSON.stringify(output,null,2));console.log('V2 baseline: four orders, RT phase, zero-ISI defect, playback/missing defects, inclusive shuffle, first-match key ranges captured.');})();
