import {canvasFont} from './fonts.js';
import {dark,tint} from './theme.js';
export function setupMonitor(post,isClosed){
  const $=id=>document.getElementById(id);let latest=null,pending=false,visible=false,lastKey='';
  const panel=$('monitorPanel'),home=$('monitorHome'),dialog=$('monitorDialog');
  function select(view){visible=view==='monitor';document.querySelector('.analysis').dataset.view=view;panel.hidden=!visible;
    for(const b of document.querySelectorAll('[data-analysis]')){b.classList.toggle('selected',b.dataset.analysis===view);b.setAttribute('aria-pressed',b.dataset.analysis===view);}
    if(visible){poll();draw();}
  }
  document.querySelectorAll('[data-analysis]').forEach(b=>b.onclick=()=>select(b.dataset.analysis));
  $('monitorCollapse').onclick=()=>select('acoustics');
  $('monitorExpand').onclick=()=>{dialog.querySelector('.monitor-dialog-body').append(panel);dialog.showModal();draw();};
  $('monitorClose').onclick=()=>dialog.close();dialog.addEventListener('close',()=>{home.append(panel);draw();});
  $('monitorStop').onclick=async()=>{if(!$('stopFrames').disabled)$('stopFrames').click();try{await post('deactivate');}catch(e){$('monitorStatus').textContent='停止失败：'+e.message;}};
  for(const id of ['monitorSeconds','monitorWindow','monitorHop'])$(id).onchange=()=>{lastKey='';poll();};
  function context(id){const canvas=$(id),w=canvas.clientWidth,h=canvas.clientHeight,d=Math.min(devicePixelRatio,2);canvas.width=w*d;canvas.height=h*d;const c=canvas.getContext('2d');c.scale(d,d);c.font=canvasFont(9);return {c,w,h};}
  function draw(){
    if(!visible)return;
    const wave=context('waveCanvas'),spec=context('spectrogramCanvas');if(!wave.w)return;
    const seconds=latest?.seconds||+$('monitorSeconds').value,end=Math.max(seconds,latest?.end||0),start=end-seconds;
    for(const {c,w,h} of [wave,spec]){c.fillStyle=tint('#fffef9');c.fillRect(0,0,w,h);c.fillStyle=tint('#7b877a');for(let i=0;i<=4;i++){const x=36+i*(w-46)/4;c.fillText((start+seconds*i/4).toFixed(1),x-5,h-3);}}
    const {c,w,h}=wave,yy=v=>8+(1-v)/2*(h-29),xx=t=>36+(t-start)/seconds*(w-46);
    c.strokeStyle=tint('#e0e4d9');for(const v of [-1,0,1]){c.beginPath();c.moveTo(36,yy(v));c.lineTo(w-10,yy(v));c.stroke();c.fillText(v,9,yy(v)+3);}
    if(latest?.waveform.length){c.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--waveform-color').trim()||tint('#386a5e');c.beginPath();latest.waveform.forEach(([lo,hi],i)=>{const x=xx(latest.end-latest.duration+(i+.5)/latest.waveform.length*latest.duration);c.moveTo(x,yy(lo));c.lineTo(x,yy(hi));});c.stroke();}
    const s=spec;for(const f of [0,2000,4000,6000])s.c.fillText(f?f/1000+'k':'0',7,8+(1-f/6000)*(s.h-30)+3);
    const data=latest?.spectrogram;
    if(data?.length){
      const cols=data.length,rows=data[0].length,img=new ImageData(cols,rows);
      const night=dark(),background=tint('#fffef9').match(/[\da-f]{2}/gi).map(v=>parseInt(v,16));
      for(let t=0;t<cols;t++)for(let f=0;f<rows;f++){const k=((rows-1-f)*cols+t)*4,v=data[t][f];
        for(let channel=0;channel<3;channel++)img.data[k+channel]=night?background[channel]+(230-background[channel])*v/255:255-v;
        img.data[k+3]=255;}
      const buffer=document.createElement('canvas');buffer.width=cols;buffer.height=rows;buffer.getContext('2d').putImageData(img,0,0);
      const x=36+(latest.end-latest.duration+latest.frame_offset-latest.hop_seconds/2-start)/seconds*(s.w-46);
      const topFrequency=rows*latest.frequency_step,plotHeight=s.h-30;
      s.c.save();s.c.beginPath();s.c.rect(36,8,s.w-46,plotHeight);s.c.clip();s.c.imageSmoothingEnabled=false;s.c.drawImage(buffer,x,8+plotHeight*(1-topFrequency/6000),cols*latest.hop_seconds/seconds*(s.w-46),plotHeight*topFrequency/6000);s.c.restore();
    }else{s.c.fillStyle=tint('#839080');s.c.fillText('点击发声后显示实际输出音频',45,s.h/2);}
  }
  async function poll(){if(pending||!visible||isClosed()||document.hidden)return;pending=true;
    try{const s=await(await post('audio/monitor',{seconds:+$('monitorSeconds').value,window_ms:+$('monitorWindow').value,hop_ms:+$('monitorHop').value})).json();
      const key=[s.generation,s.end,s.seconds,s.window_ms,s.hop_seconds].join(':');latest=s;
      $('monitorStatus').textContent=`48 kHz 输出 · 分析窗 ${s.window_ms} ms（1/T ≈ ${Math.round(s.resolution_hz)} Hz）· 实际步长 ${(s.hop_seconds*1000).toFixed(1)} ms · 90 dB`;
      if(key!==lastKey){lastKey=key;draw();}
    }catch(e){$('monitorStatus').textContent='分析未更新：'+e.message;}finally{pending=false;}
  }
  document.addEventListener('m10-theme',draw);
  new ResizeObserver(draw).observe(panel);setInterval(poll,180);select('acoustics');
}
