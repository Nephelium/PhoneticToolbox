// Synthetic frame IDs and scheduled audio pulses. No physical devices.
const fs=require('node:fs/promises'),path=require('node:path'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/m05-r3/clock-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/m05-probe.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true}),page=await browser.newPage();
 try{
 await page.goto(server.resolvedUrls.local[0]+'tests/m05-probe.html');
 const data=await page.evaluate(async config=>{
  const {LipCapture}=await import('/src/modules/lip-extraction/capture.ts'),{LipInference}=await import('/src/modules/lip-extraction/inference.ts');
  const samples=[],events=[];LipInference.prototype.initialize=async()=>{await new Promise(r=>setTimeout(r,config.initializeMs));return {test:true};};
  LipInference.prototype.detect=async function(image,time_ms){const c=new OffscreenCanvas(64,64),x=c.getContext('2d');x.drawImage(image,0,0,64,64);let code=0;for(let bit=0;bit<8;bit++)if(x.getImageData(bit*8+4,32,1,1).data[0]>128)code|=1<<bit;samples.push({time_ms,id:code,at:performance.now()});image.close();await new Promise(r=>setTimeout(r,config.inferenceMs));return {points:null,metrics:null,inference_ms:config.inferenceMs,width:64,height:64};};
  const canvas=document.createElement('canvas');canvas.width=canvas.height=64;const cx=canvas.getContext('2d');let id=0;
  const audio=new AudioContext({sampleRate:48000}),dest=audio.createMediaStreamDestination(),osc=audio.createOscillator(),gain=audio.createGain();osc.connect(gain);gain.connect(dest);osc.start();gain.gain.value=0;await audio.resume();
  const draw=()=>{id=(id+1)%240;for(let bit=0;bit<8;bit++){cx.fillStyle=id&(1<<bit)?'white':'black';cx.fillRect(bit*8,0,8,64);}events.push({id,at:performance.now(),audio:audio.currentTime});};draw();const timer=setInterval(draw,33);
  const stream=canvas.captureStream(30);stream.addTrack(dest.stream.getAudioTracks()[0]);const video=document.createElement('video');video.muted=true;document.body.append(video);
  const cap=new LipCapture(video,()=>{},async()=>stream);
  await cap.start({mode:'realtime',camera:'',microphone:'explicit-test',requestedFps:30,filter:false,cutoff:15,delegate:'CPU'});
  const onset=audio.currentTime+.6;gain.gain.setValueAtTime(.4,onset);gain.gain.setValueAtTime(0,onset+.05);
  await new Promise(r=>setTimeout(r,2500));await cap.stop();clearInterval(timer);const meta=cap.metadata(),blob=cap.blob();const raw=new Uint8Array(await blob.arrayBuffer());let binary='';for(let i=0;i<raw.length;i+=8192)binary+=String.fromCharCode(...raw.subarray(i,i+8192));cap.dispose();osc.stop();await audio.close();
  return {config,meta,frames:cap.frames,samples,events,onset,media:btoa(binary)};
 },{initializeMs:Number(process.env.M05_CLOCK_INIT_MS||0),inferenceMs:Number(process.env.M05_CLOCK_INFER_MS||37)});
 await fs.writeFile(path.join(out,'source.webm'),Buffer.from(data.media,'base64'));delete data.media;await fs.writeFile(path.join(out,'capture.json'),JSON.stringify(data,null,2));console.log(out);
 }finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1});
