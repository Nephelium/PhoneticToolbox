import {canvasFont,freezeFonts,svgFontFamily} from './fonts.js';
import {VocalTractViewer} from './scene.js';
import {frameIntervals,silentAt} from './keyframes.js';
import {dark} from './theme.js';

const b64=bytes=>{let s='';for(let i=0;i<bytes.length;i+=8192)s+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(s);};
const un64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));

function timeline(c,width,height,time,frames,curve){
  const intervals=frameIntervals(frames),total=intervals.at(-1).end;
  const panel=dark()?'#182432':'#ffffff',ink=dark()?'#e9f0f9':'#132a49',line=dark()?'#304154':'#dce5f0',green=dark()?'#7cdad4':'#00787e';
  const top=height-156,left=60,right=width-24,at=t=>left+t/total*(right-left),fy=f=>top+54+(350-f)/290*72;
  c.fillStyle=panel;c.fillRect(0,top,width,156);c.strokeStyle=line;c.strokeRect(0,top,width,156);
  c.font=canvasFont(16);c.textAlign='left';c.fillStyle=ink;
  const current=intervals.find(f=>time<f.end)||intervals.at(-1);
  c.font=canvasFont(16,true);c.fillText(current.name,16,top+23);const nameWidth=c.measureText(current.name).width;
  c.font=canvasFont(16);c.fillText(`  ·  ${current.start.toFixed(2)}–${current.end.toFixed(2)} s`,16+nameWidth,top+23);
  c.textAlign='right';c.fillText(`${time.toFixed(2)} / ${total.toFixed(2)} s  ·  PhoneticToolbox / VTL 2.4`,width-20,top+23);
  function value(t){
    if(curve.length){let i=1;while(i<curve.length-1&&curve[i][0]<t/total)i++;const a=curve[i-1],b=curve[i],u=(t/total-a[0])/(b[0]-a[0]);return a[1]+u*(b[1]-a[1]);}
    const f=intervals.find(f=>t<f.end)||intervals.at(-1),a=frames[f.index],b=frames[f.index+1]||a;
    let u=Math.max(0,Math.min(1,(t-f.start)/(f.end-f.start)));u=u*u*(3-2*u);return a.f0*(1-u)+(b.silent?a.f0:b.f0)*u;
  }
  c.font=canvasFont(11);c.textAlign='right';
  for(const f of [60,150,250,350]){c.strokeStyle=line;c.beginPath();c.moveTo(left,fy(f));c.lineTo(right,fy(f));c.stroke();c.fillStyle=ink;c.fillText(String(f),left-8,fy(f)+4);}
  c.textAlign='center';
  for(const f of intervals){const x=at(f.start),end=at(f.end);c.fillStyle=f.index===current.index?green+'20':line+'35';c.fillRect(x,top+32,end-x,103);c.strokeStyle=line;c.beginPath();c.moveTo(x,top+32);c.lineTo(x,top+135);c.stroke();c.save();c.beginPath();c.rect(x+2,top+32,Math.max(0,end-x-4),20);c.clip();c.fillStyle=ink;c.font=canvasFont(11,true);c.fillText(f.name,(x+end)/2,top+46);c.restore();}
  c.strokeStyle=green;c.lineWidth=2;c.beginPath();let pen=false;for(let i=0;i<=400;i++){if(silentAt(frames,i/400)){pen=false;continue;}const x=at(total*i/400),y=fy(value(total*i/400));pen?c.lineTo(x,y):c.moveTo(x,y);pen=true;}c.stroke();
  c.strokeStyle=dark()?'#ffca8c':'#ac5825';c.beginPath();c.moveTo(at(time),top+31);c.lineTo(at(time),top+135);c.stroke();
  c.textAlign='left';c.fillStyle=ink;const voiced=(frames[current.index].source?.vibration??1)>0;c.fillText(current.silent?'静音 · 无基频':voiced?`F0 ${Math.round(value(time))} Hz`:'无周期基频（耳语 / 清声）',16,top+150);
  c.textAlign='center';for(let i=1;i<=4;i++)c.fillText((total*i/4).toFixed(2)+' s',at(total*i/4),top+150);
}

export function setupVideo({post,viewer,getFrames,getCurve,prepare,lock,ready}){
  const $=id=>document.getElementById(id),dialog=$('videoDialog');let running=false,cancel=false;
  const status=s=>$('videoStatus').textContent=s;
  $('exportVideo').onclick=()=>{status('视频包含同步音频、构形动作、姿势名称和基频。');$('videoProgress').value=0;dialog.showModal();};
  $('videoClose').onclick=()=>{if(running){cancel=true;status('正在取消…');post('animation/stop').catch(()=>{});}else dialog.close();};
  dialog.addEventListener('cancel',e=>{if(running){e.preventDefault();$('videoClose').click();}});
  $('videoStart').onclick=async()=>{
    if(running||!ready())return;
    running=true;cancel=false;lock(true);$('motionHud').hidden=true;$('videoStart').disabled=true;$('videoViews').disabled=true;$('videoClose').textContent='取消导出';
    let session=null,exportViewer=null,encoder=null,audioEncoder=null;let releaseFonts=()=>{};
    try{
      releaseFonts=await freezeFonts();
      const six=$('videoViews').value==='six',width=six?1920:1280,height=six?1080:720,fps=30;
      const videoConfig={codec:'vp8',width,height,bitrate:six?8000000:4000000,framerate:fps,latencyMode:'quality'};
      const audioConfig={codec:'opus',sampleRate:48000,numberOfChannels:1,bitrate:96000};
      if(!window.VideoEncoder||!window.AudioEncoder||!(await VideoEncoder.isConfigSupported(videoConfig)).supported||!(await AudioEncoder.isConfigSupported(audioConfig)).supported)throw Error('当前桌面运行时不支持视频编码，请使用新版 Windows 应用');
      const duration=Math.round(getFrames().reduce((s,f)=>s+f.duration,0)*48000)/48000;
      if(cancel)throw Error('已取消导出');
      session=await(await post('video/begin',{width,height,fps,duration})).json();if(session.cancelled)return;
      status('正在准备动作与音频…');const prepared=await prepare();if(cancel)throw Error('已取消导出');
      const audio=await(await post('animation/audio',{id:prepared.id})).json();
      const tileW=six?width/3:width,tileH=six?(height-156)/2:height-156;
      const host=document.createElement('div');host.className='export-viewport';host.style.cssText=`position:fixed;left:-10000px;top:0;width:${tileW}px;height:${tileH}px;overflow:hidden`;document.body.append(host);
      exportViewer=new VocalTractViewer(host,()=>{});exportViewer.metadata=viewer.metadata;await exportViewer.load();
      exportViewer.setOptions({head:viewer.showHead,nose:viewer.showNose,fullModel:viewer.fullModel,teeth:viewer.showTeeth,labels:viewer.labels,focused:viewer.focused});
      exportViewer.zoom=viewer.zoom;exportViewer.pan=[...viewer.pan];exportViewer.camera.copy(viewer.camera);exportViewer.camera.aspect=tileW/tileH;exportViewer.camera.updateProjectionMatrix();exportViewer.orbit.target.copy(viewer.orbit.target);exportViewer.orbit.enableDamping=false;exportViewer.orbit.enabled=false;
      exportViewer.renderer.setPixelRatio(1);exportViewer.width=0;exportViewer.resize();
      const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;const c=canvas.getContext('2d',{alpha:false});
      let pending=[],opus=null,failed=null;
      const output=track=>(chunk,meta)=>{const bytes=new Uint8Array(chunk.byteLength);chunk.copyTo(bytes);pending.push({track,timestamp:chunk.timestamp,duration:chunk.duration||0,key:chunk.type==='key',data:b64(bytes)});if(track===2&&meta.decoderConfig?.description)opus=b64(new Uint8Array(meta.decoderConfig.description));};
      encoder=new VideoEncoder({output:output(1),error:e=>{failed=e;}});encoder.configure(videoConfig);
      audioEncoder=new AudioEncoder({output:output(2),error:e=>{failed=e;}});audioEncoder.configure(audioConfig);
      async function flush(){while(pending.length){if(cancel)throw Error('已取消导出');const batch=pending.splice(0,20);await post('video/chunks',{id:session.id,packets:batch,...(opus?{opus}:{})});}if(failed)throw failed;}
      const samples=new Float32Array(un64(audio.base64).buffer);
      for(let offset=0;offset<samples.length;offset+=960){
        const data=new AudioData({format:'f32-planar',sampleRate:audio.sample_rate,numberOfChannels:1,numberOfFrames:Math.min(960,samples.length-offset),timestamp:Math.round(offset/audio.sample_rate*1e6),data:samples.subarray(offset,offset+960)});
        audioEncoder.encode(data);data.close();if(offset%9600===0){await new Promise(resolve=>setTimeout(resolve,0));await flush();}
      }
      await audioEncoder.flush();await flush();
      const views=six?[['organs',true],['airway',true],['overlay',true],['organs',false],['airway',false],['overlay',false]]:[[viewer.mode,viewer.sagittal]];
      const frameCount=Math.ceil(audio.samples*fps/audio.sample_rate);
      const modes={organs:'器官观察',airway:'气腔观察',overlay:'叠加观察'};
      for(let i=0;i<frameCount;i++){
        if(cancel)throw Error('已取消导出');const time=i/fps;
        const picture=await(await post('animation/picture',{id:prepared.id,index:i})).json();
        c.fillStyle=dark()?'#192d39':'#f5f3eb';c.fillRect(0,0,width,height);
        for(let v=0;v<views.length;v++){
          const [mode,sagittal]=views[v],x=six?v%3*tileW:0,y=six?Math.floor(v/3)*tileH:0;
          exportViewer.sagittal=sagittal;exportViewer.mode=mode;exportViewer.update(picture.state);exportViewer.resize();
          if(sagittal){
            const svg=exportViewer.svg.cloneNode(true);svg.setAttribute('width',tileW);svg.setAttribute('height',tileH);svg.setAttribute('xmlns','http://www.w3.org/2000/svg');svg.setAttribute('font-family',svgFontFamily());
            const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)],{type:'image/svg+xml'}));
            try{const img=new Image();img.src=url;await img.decode();c.drawImage(img,x,y,tileW,tileH);}finally{URL.revokeObjectURL(url);}
          }else{
            exportViewer.renderNow();c.drawImage(exportViewer.renderer.domElement,x,y,tileW,tileH);
            if(viewer.labels){c.font=canvasFont(13);c.textAlign='left';c.fillStyle=dark()?'#d7f3eb':'#286657';for(const {label} of exportViewer.controls.values())if(!label.hidden)c.fillText(label.textContent,x+parseFloat(label.style.left),y+parseFloat(label.style.top)+12);}
          }
          c.fillStyle=dark()?'#d5e8ef':'#274957';c.font=canvasFont(16);c.textAlign='left';c.fillText(`${sagittal?'正中矢状面':'三维'} · ${modes[mode]}`,x+14,y+25);c.strokeStyle=dark()?'#304154':'#dce5f0';c.lineWidth=1;c.strokeRect(x,y,tileW,tileH);
        }
        timeline(c,width,height,time,prepared.frames,getCurve());
        const frame=new VideoFrame(canvas,{timestamp:Math.round(i*1e6/fps),duration:Math.round(Math.min(1/fps,prepared.duration-time)*1e6)});encoder.encode(frame,{keyFrame:i%fps===0});frame.close();
        if(i%6===0){await encoder.flush();await flush();}
        $('videoProgress').value=(i+1)/frameCount;status(`正在导出 ${i+1} / ${frameCount} 帧${prepared.cached?' · 复用已有动作与音频':''}`);
      }
      await encoder.flush();await flush();if(cancel)throw Error('已取消导出');
      const saved=await(await post('video/finish',{id:session.id})).json();status(`已保存 ${saved.name} · ${saved.frames} 帧 · ${saved.duration.toFixed(2)} 秒`);session=null;
    }catch(e){status(cancel?'已取消导出，未替换目标文件。':'导出失败：'+e.message);}
    finally{releaseFonts();
      if(session?.id)await post('video/cancel',{id:session.id}).catch(()=>{});
      if(encoder&&encoder.state!=='closed')encoder.close();if(audioEncoder&&audioEncoder.state!=='closed')audioEncoder.close();exportViewer?.dispose();
      if(running){running=false;lock(false);}$('videoStart').disabled=false;$('videoViews').disabled=false;$('videoClose').textContent='返回工作台';
    }
  };
}
