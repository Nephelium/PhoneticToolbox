import {setupPitch} from './pitch.js';
export function setupKeyframes({initialFrames=[],initialCurve=[],getPose,applyPose,post,setBusy,isReady,isSilent,keepVowel}){
  const $=id=>document.getElementById(id);let frames=initialFrames,busy=false,generation=0,selected=-1,page=0,saveQueue=Promise.resolve();
  const pageSize=()=>document.getElementById('pitchEditor').hidden?3:2;
  const pitch=setupPitch({initialCurve,getFrames:()=>frames,isBusy:()=>busy,onChange:()=>save()});
  const save=()=>{const captured=JSON.parse(JSON.stringify(frames)),pitch_curve=pitch.getCurve();saveQueue=saveQueue.catch(()=>{}).then(()=>post('keyframes',{frames:captured,pitch_curve})).catch(e=>{text('保存失败：'+e.message);throw e;});saveQueue.catch(()=>{});};
  const text=message=>$('frameStatus').textContent=message;
  function render(){
    $('keyframeList').replaceChildren();
    page=Math.max(0,Math.min(page,Math.ceil(frames.length/pageSize())-1));
    frames.forEach((f,i)=>{if(Math.floor(i/pageSize())!==page)return;
      const card=document.createElement('div');card.className='pose-card'+(selected===i?' current':'');
      const title=document.createElement('button');title.className='pose-title';title.textContent=`${i+1} · ${f.preset?'/'+({E:'ɛ','2':'ø'}[f.preset]||f.preset)+'/':'自定义姿势'}`;title.disabled=busy;
      title.onclick=()=>{selected=i;applyPose(f);render();};
      const detail=document.createElement('small');detail.textContent=`唇宽 ${Math.round(f.lip_width*100)}% · ${{voiceless:'清声',whisper:'耳语近似',transition:'模式过渡'}[f.source?.mode]||Math.round(f.f0)+' Hz'}`;
      const label=document.createElement('label');label.textContent=i===frames.length-1?'停留':'过渡';
      const duration=document.createElement('input');Object.assign(duration,{type:'number',min:.15,max:3,step:.05,value:f.duration,disabled:busy});duration.setAttribute('aria-label',`姿势 ${i+1} ${i===frames.length-1?'停留':'过渡'}秒数`);
      duration.onchange=()=>{const n=+duration.value;if(Number.isFinite(n)&&n>=.15&&n<=3){f.duration=n;save();}else duration.value=f.duration;render();};label.append(duration,document.createTextNode('秒'));
      const tools=document.createElement('div');tools.className='pose-tools';
      for(const [name,fn,disabled] of [['←',()=>{[frames[i-1],frames[i]]=[frames[i],frames[i-1]];selected=i-1;},i===0],['→',()=>{[frames[i+1],frames[i]]=[frames[i],frames[i+1]];selected=i+1;},i===frames.length-1],['移除',()=>{frames.splice(i,1);selected=-1;},false]]){
        const b=document.createElement('button');b.textContent=name;b.disabled=busy||disabled;b.setAttribute('aria-label',`${name==='←'?'前移':name==='→'?'后移':'移除'}姿势 ${i+1}`);b.onclick=()=>{fn();save();render();};tools.append(b);
      }
      card.append(title,detail,label,tools);$('keyframeList').append(card);
    });
    const total=frames.reduce((s,f)=>s+f.duration,0);
    $('frameCount').textContent=frames.length;$('transportStatus').textContent=frames.length?`${frames.length} 个姿势 · ${total.toFixed(2)} 秒`:'保存多个姿势后连续播放';
    $('framePage').textContent=`${page+1} / ${Math.max(1,Math.ceil(frames.length/pageSize()))}`;$('framePrev').disabled=page===0;$('frameNext').disabled=(page+1)*pageSize()>=frames.length;
    $('captureFrame').disabled=busy||frames.length>=12;$('playFrames').disabled=busy||isSilent||frames.length<2||total>12;$('stopFrames').disabled=!busy;
    pitch.draw();
    if(!busy)text(frames.length?`${frames.length} 个姿势 · ${total.toFixed(2)} 秒${total>12?'，请缩短到 12 秒以内':''}`:'先摆出姿势，再保存；至少需要两个关键帧。');
  }
  function lock(value){busy=value;setBusy(value);$('motionHud').hidden=!value;render();}
  $('captureFrame').onclick=()=>{if(!isReady()){text('正在更新构形，稍后再保存。');return;}const pose=getPose();frames.push({...pose,params:[...pose.params],duration:.6});selected=frames.length-1;page=Math.floor(selected/pageSize());save();render();};
  $('stopFrames').onclick=async()=>{++generation;try{await post('animation/stop');}catch{}lock(false);text('已停止，保留当前姿势。');};
  $('stopMotionOverlay').onclick=()=>$('stopFrames').click();
  $('playFrames').onclick=async()=>{
    if(!isReady()){text('正在更新构形，稍后再播放。');return;}
    const run=++generation;lock(true);$('motionClock').textContent='正在生成声音与画面';text('正在生成声音与同步画面…');$('motionProgress').value=0;
    try{
      const prepared=await(await post('animation/prepare',{frames,pitch_curve:pitch.getCurve(),keep_vowel:keepVowel()})).json();if(run!==generation)return;
      const adjusted=JSON.stringify(prepared.frames.map(f=>f.params))!==JSON.stringify(frames.map(f=>f.params));
      if(adjusted){frames=prepared.frames;save();render();}
      await post('animation/play',{id:prepared.id});if(run!==generation){await post('animation/stop');return;}
      let lastIndex=-1;
      const tick=async()=>{
        if(run!==generation)return;
        try{
          const frame=await(await post('animation/frame',{id:prepared.id})).json();if(run!==generation)return;
          if(frame.index!==lastIndex){lastIndex=frame.index;applyPose(frame.state,true);}
          $('motionProgress').value=frame.time/frame.duration;$('motionClock').textContent=`${frame.time.toFixed(2)} / ${frame.duration.toFixed(2)} 秒`;text(`${frame.time.toFixed(2)} / ${frame.duration.toFixed(2)} 秒${adjusted?' · 已保护端点口腔通路':''}`);
          if(frame.audio_error)throw Error(frame.audio_error);
          if(frame.active)setTimeout(tick,40);
          else{lock(false);text(frame.completed?'播放结束 · 已保留最后一个姿势':'已停止');}
        }catch(e){await post('animation/stop').catch(()=>{});lock(false);text('播放失败：'+e.message);}
      };await tick();
    }catch(e){if(run===generation){lock(false);text('生成失败：'+e.message);}}
  };
  $('framePrev').onclick=()=>{page--;render();};$('frameNext').onclick=()=>{page++;render();};
  document.addEventListener('pitch-layout',render);
  render();return {isBusy:()=>busy};
}
