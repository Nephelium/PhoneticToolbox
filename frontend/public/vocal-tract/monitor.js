import {canvasFont} from './fonts.js';
import {tint} from './theme.js';
import {zoomMonitor,monitorRange,monitorTimeLabel,waveformLimit} from './monitor-view.mjs';
export function setupMonitor(post,isClosed){
  const $=id=>document.getElementById(id);let latest=null,pending=false,visible=false,lastKey='',active=true,outputGeneration,requestKey='';
  document.addEventListener('m10-output-state',event=>{if(active&&!event.detail.active)requestKey='';active=event.detail.active;outputGeneration=event.detail.generation;});
  const panel=$('monitorPanel'),home=$('monitorHome'),dialog=$('monitorDialog');
  let view={span:+$('monitorSeconds').value,lag:0};
  function resetZoom(){view={span:+$('monitorSeconds').value,lag:0};lastKey='';draw();poll();}
  for(const id of ['waveCanvas','spectrogramCanvas']){
    const canvas=$(id);canvas.title='Ctrl＋滚轮缩放时间；双击恢复显示时长';
    canvas.addEventListener('wheel',e=>{if(!e.ctrlKey)return;e.preventDefault();e.stopPropagation();
      const rect=canvas.getBoundingClientRect(),delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?rect.height:1);
      view=zoomMonitor(view,delta,(e.clientX-rect.left-36)/(rect.width-46));draw();lastKey='';poll();
    },{passive:false});canvas.addEventListener('dblclick',resetZoom);
  }
  function select(view){visible=view==='monitor';document.querySelector('.analysis').dataset.view=view;panel.hidden=!visible;
    for(const b of document.querySelectorAll('[data-analysis]')){b.classList.toggle('selected',b.dataset.analysis===view);b.setAttribute('aria-pressed',b.dataset.analysis===view);}
    if(visible){poll();draw();}
  }
  document.querySelectorAll('[data-analysis]').forEach(b=>b.onclick=()=>select(b.dataset.analysis));
  $('monitorCollapse').onclick=()=>select('acoustics');
  $('monitorExpand').onclick=()=>{dialog.querySelector('.monitor-dialog-body').append(panel);dialog.showModal();draw();};
  $('monitorClose').onclick=()=>dialog.close();dialog.addEventListener('close',()=>{home.append(panel);draw();});
  $('monitorStop').onclick=async()=>{if(!$('stopFrames').disabled)$('stopFrames').click();try{await post('deactivate');}catch(e){$('monitorStatus').textContent='停止失败：'+e.message;}};
  $('monitorSeconds').onchange=resetZoom;
  $('monitorAmplitude').onchange=draw;
  for(const id of ['monitorWindow','monitorHop'])$(id).onchange=()=>{lastKey='';poll();};
  function context(id){const canvas=$(id),w=canvas.clientWidth,h=canvas.clientHeight,d=Math.min(devicePixelRatio,2);canvas.width=w*d;canvas.height=h*d;const c=canvas.getContext('2d');c.scale(d,d);c.font=canvasFont(9);return {c,w,h};}
  function draw(){
    if(!visible||panel.clientWidth<=0||panel.clientHeight<=0)return;
    const wave=context('waveCanvas'),spec=context('spectrogramCanvas');if(!wave.w)return;
    const {seconds,start,end}=monitorRange(view,latest?.end||0);
    $('waveLabel').textContent='波形 · 振幅 · '+(seconds<1?(seconds*1000).toFixed(seconds<.01?1:0)+' ms':seconds.toFixed(2)+' s');
    for(const id of ['waveCanvas','spectrogramCanvas'])Object.assign($(id).dataset,{viewStart:String(start),viewEnd:String(end),viewSeconds:String(seconds)});
    for(const {c,w,h} of [wave,spec]){
      c.fillStyle=tint('#fffef9');c.fillRect(0,0,w,h);c.fillStyle=tint('#7b877a');
      const labelWidth=Math.max(c.measureText(monitorTimeLabel(start,seconds)).width,c.measureText(monitorTimeLabel(end,seconds)).width);
      const ticks=Math.max(1,Math.min(4,Math.floor((w-46)/(labelWidth+12))-1));
      for(let i=0;i<=ticks;i++){const x=36+i*(w-46)/ticks;c.textAlign=i===0?'left':i===ticks?'right':'center';c.fillText(monitorTimeLabel(start+seconds*i/ticks,seconds),x,h-3);}c.textAlign='left';
    }
    const limit=waveformLimit(latest?.waveform_peak,$('monitorAmplitude').value==='auto');
    const {c,w,h}=wave,yy=v=>8+(1-v/limit)/2*(h-29),xx=t=>36+(t-start)/seconds*(w-46);
    c.strokeStyle=tint('#e0e4d9');for(const v of [-limit,0,limit]){c.beginPath();c.moveTo(36,yy(v));c.lineTo(w-10,yy(v));c.stroke();c.fillText(Number(v.toPrecision(2)).toString(),2,yy(v)+3);}
    const raw=latest?.waveform_samples||[],envelope=latest?.waveform||[];
    Object.assign($('waveCanvas').dataset,{waveMode:latest?.waveform_mode||'envelope',sampleCount:String(raw.length),amplitudeLimit:String(limit)});
    if(raw.length||envelope.length){
      c.save();c.beginPath();c.rect(36,8,w-46,h-29);c.clip();c.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--waveform-color').trim()||tint('#386a5e');c.beginPath();
      if(latest.waveform_mode==='samples')raw.forEach((v,i)=>{const x=xx(latest.waveform_start+i/latest.sample_rate);if(i===0)c.moveTo(x,yy(v));else c.lineTo(x,yy(v));});
      else envelope.forEach(([lo,hi],i)=>{const x=xx(latest.waveform_start+(i+.5)/envelope.length*latest.waveform_duration);c.moveTo(x,yy(lo));c.lineTo(x,yy(hi));});
      c.stroke();c.restore();
    }
    const s=spec;for(const f of [0,2000,4000,6000])s.c.fillText(f?f/1000+'k':'0',7,8+(1-f/6000)*(s.h-30)+3);
    const data=latest?.spectrogram;
    if(data?.length){
      const cols=data.length,rows=data[0].length,img=new ImageData(cols,rows);
      const background=tint('#fffef9').match(/[\da-f]{2}/gi).map(v=>parseInt(v,16)),foreground=tint('#297c6b').match(/[\da-f]{2}/gi).map(v=>parseInt(v,16));
      for(let t=0;t<cols;t++)for(let f=0;f<rows;f++){const k=((rows-1-f)*cols+t)*4,v=data[t][f];
        for(let channel=0;channel<3;channel++)img.data[k+channel]=background[channel]+(foreground[channel]-background[channel])*v/255;
        img.data[k+3]=255;}
      const buffer=document.createElement('canvas');buffer.width=cols;buffer.height=rows;buffer.getContext('2d').putImageData(img,0,0);
      const x=36+(latest.end-latest.duration+latest.frame_offset-latest.hop_seconds/2-start)/seconds*(s.w-46);
      const topFrequency=rows*latest.frequency_step,plotHeight=s.h-30;
      s.c.save();s.c.beginPath();s.c.rect(36,8,s.w-46,plotHeight);s.c.clip();s.c.imageSmoothingEnabled=false;s.c.drawImage(buffer,x,8+plotHeight*(1-topFrequency/6000),cols*latest.hop_seconds/seconds*(s.w-46),plotHeight*topFrequency/6000);s.c.restore();
    }else{s.c.fillStyle=tint('#839080');s.c.fillText('点击发声后显示实际输出音频',45,s.h/2);}
  }
  async function poll(){if(pending||!visible||isClosed()||document.hidden||panel.clientWidth<=0||panel.clientHeight<=0)return;pending=true;
    try{const requested={...view},width=Math.max(64,Math.min(2048,Math.ceil($('waveCanvas').clientWidth-46)));
      const keyBefore=[requested.span,requested.lag,width,$('monitorWindow').value,outputGeneration].join(':');
      if(!active&&latest&&requestKey===keyBefore&&lastKey)return;
      const s=await(await post('audio/monitor',{seconds:Math.max(1,Math.min(12,requested.span+requested.lag)),window_ms:+$('monitorWindow').value,hop_ms:+$('monitorHop').value,view_seconds:requested.span,view_lag:requested.lag,waveform_width:width})).json();
      if(requested.span!==view.span||requested.lag!==view.lag)return;
      requestKey=keyBefore;
      const key=[s.generation,s.end,s.seconds,s.window_ms,s.hop_seconds,requested.span,requested.lag,width].join(':');latest=s;
      $('monitorStatus').textContent=`48 kHz 输出 · 分析窗 ${s.window_ms} ms（1/T ≈ ${Math.round(s.resolution_hz)} Hz）· 实际步长 ${(s.hop_seconds*1000).toFixed(1)} ms · 90 dB`;
      if(key!==lastKey){lastKey=key;draw();}
    }catch(e){$('monitorStatus').textContent='分析未更新：'+e.message;}finally{pending=false;}
  }
  document.addEventListener('m10-theme',draw);
  new ResizeObserver(draw).observe(panel);setInterval(poll,180);select('acoustics');
}
