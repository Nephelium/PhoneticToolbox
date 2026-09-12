import {setupPitch} from './pitch.js';
export const frameName=f=>f.name|| (f.silent?'静音':f.preset?'/'+({E:'ɛ','2':'ø'}[f.preset]||f.preset)+'/':'自定义姿势');
export function frameIntervals(frames){let time=0;return frames.map((f,index)=>{const start=time;time+=f.duration;return {index,name:frameName(f),start,end:time,silent:!!f.silent};});}
export function silentAt(frames,x){const intervals=frameIntervals(frames),time=x*(intervals.at(-1)?.end||0);return !!(intervals.find(f=>time<f.end)||intervals.at(-1))?.silent;}
export function setupKeyframes({initialFrames=[],initialCurve=[],getPose,applyPose,post,setBusy,isReady,isSilent,keepVowel,viewer}){
  const $=id=>document.getElementById(id);let frames=initialFrames,busy=false,generation=0,selected=-1,editing=-1,saveQueue=Promise.resolve(),saveVersion=0,savedVersion=0;
  let clearedSnapshot=null,dragIndex=-1,dropIndex=-1,dragY=0,dragScroll=0,suppressClick=false;
  const list=$('keyframeList');
  const dirty=value=>parent.postMessage({type:'m10-dirty',dirty:value},'*');
  const pitch=setupPitch({initialCurve,getFrames:()=>frames,isBusy:()=>busy,onChange:()=>save()});
  const syncDirty=()=>dirty(editing>=0||savedVersion<saveVersion);
  const save=(preserveClear=false)=>{if(!preserveClear){clearedSnapshot=null;$('undoClearFrames').hidden=true;}const captured=JSON.parse(JSON.stringify(frames)),pitch_curve=pitch.getCurve(),version=++saveVersion;syncDirty();saveQueue=saveQueue.catch(()=>{}).then(()=>post('keyframes',{frames:captured,pitch_curve})).then(()=>{savedVersion=version;syncDirty();}).catch(e=>{text('保存失败：'+e.message);throw e;});saveQueue.catch(()=>{});return saveQueue;};
  const text=message=>$('frameStatus').textContent=message;
  function clearDrag(){
    cancelAnimationFrame(dragScroll);dragScroll=0;dragIndex=-1;dropIndex=-1;
    for(const card of list.children)card.classList.remove('dragging','drop-before','drop-after');
  }
  function markDrop(){
    const cards=[...list.children];dropIndex=cards.findIndex(card=>{const r=card.getBoundingClientRect();return dragY<r.top+r.height/2;});
    if(dropIndex<0)dropIndex=cards.length;
    for(const card of cards)card.classList.remove('drop-before','drop-after');
    if(dropIndex<cards.length)cards[dropIndex].classList.add('drop-before');
    else cards.at(-1)?.classList.add('drop-after');
  }
  function scrollDrag(){
    if(dragIndex<0)return;
    const r=list.getBoundingClientRect(),edge=36;
    const delta=dragY<r.top+edge?-Math.min(14,(r.top+edge-dragY)/3):dragY>r.bottom-edge?Math.min(14,(dragY-r.bottom+edge)/3):0;
    if(delta){list.scrollTop+=delta;markDrop();}
    dragScroll=requestAnimationFrame(scrollDrag);
  }
  function moveFrame(from,to){
    if(busy||editing>=0||from===to||to<0||to>=frames.length)return;
    const current=frames[selected],moved=frames.splice(from,1)[0];frames.splice(to,0,moved);
    selected=current?frames.indexOf(current):-1;save();render();
    list.children[to]?.scrollIntoView({block:'nearest'});text(`已将 ${frameName(moved)} 移至第 ${to+1} 项`);
  }
  list.ondragover=e=>{if(dragIndex<0||busy||editing>=0)return;e.preventDefault();e.dataTransfer.dropEffect='move';dragY=e.clientY;markDrop();if(!dragScroll)dragScroll=requestAnimationFrame(scrollDrag);};
  list.ondragleave=e=>{if(!list.contains(e.relatedTarget)){cancelAnimationFrame(dragScroll);dragScroll=0;}};
  list.ondragenter=()=>{if(dragIndex>=0&&!dragScroll)dragScroll=requestAnimationFrame(scrollDrag);};
  list.ondrop=e=>{
    if(dragIndex<0)return;e.preventDefault();dragY=e.clientY;markDrop();
    const from=dragIndex,to=dropIndex-(dropIndex>from?1:0);clearDrag();moveFrame(from,to);
    setTimeout(()=>{suppressClick=false;},0);
  };
  function render(){
    clearDrag();
    const scroll=$('keyframeList').scrollTop;
    $('keyframeList').replaceChildren();
    const intervals=frameIntervals(frames);
    frames.forEach((f,i)=>{
      const card=document.createElement('div');card.className='pose-card'+(selected===i?' current':'');
      card.draggable=!busy&&editing<0;card.title='拖动卡片空白处上下排序';
      card.onpointerdown=e=>{card.draggable=!busy&&editing<0&&!e.target.closest('button,input,label');};
      card.ondragstart=e=>{
        if(busy||editing>=0||!card.draggable){e.preventDefault();return;}
        dragIndex=i;dragY=e.clientY;suppressClick=true;e.dataTransfer.effectAllowed='move';e.dataTransfer.setData('text/plain',String(i));
        card.classList.add('dragging');dragScroll=requestAnimationFrame(scrollDrag);
      };
      card.ondragend=()=>{clearDrag();setTimeout(()=>{suppressClick=false;},0);};
      card.tabIndex=0;card.setAttribute('aria-label',`载入姿势 ${i+1} ${frameName(f)}`);
      const select=()=>{if(busy)return;if(editing>=0&&editing!==i){text('请先保存或取消正在编辑的姿势。');return;}selected=i;if(editing!==i)applyPose(f);render();};
      card.onclick=e=>{if(!suppressClick&&!e.target.closest('button,input,label'))select();};
      card.onkeydown=e=>{if(e.target!==card)return;if(e.altKey&&['ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();moveFrame(i,i+(e.key==='ArrowUp'?-1:1));}else if(['Enter',' '].includes(e.key)){e.preventDefault();select();}};
      const title=document.createElement('button');title.className='pose-title';title.textContent=`${i+1} · ${frameName(f)}`;title.disabled=busy;
      title.onclick=select;
      const name=document.createElement('input');Object.assign(name,{type:'text',value:frameName(f),maxLength:80,disabled:busy});name.className='pose-name';name.setAttribute('aria-label',`姿势 ${i+1} 名称`);
      name.onchange=()=>{f.name=name.value.trim();save();render();};
      const detail=document.createElement('small');detail.textContent=f.silent?'静音 · 保持构形 · 无基频':`唇宽 ${Math.round(f.lip_width*100)}% · ${{voiceless:'清声',whisper:'耳语近似',transition:'模式过渡'}[f.source?.mode]||Math.round(f.f0)+' Hz'}`;
      const label=document.createElement('label');label.textContent=f.silent?'静音':i===frames.length-1?'停留':'过渡';
      const duration=document.createElement('input');Object.assign(duration,{type:'number',min:.05,max:3,step:.05,value:f.duration,disabled:busy});duration.setAttribute('aria-label',`姿势 ${i+1} ${f.silent?'静音':i===frames.length-1?'停留':'过渡'}秒数`);
      duration.onchange=()=>{const n=+duration.value;if(Number.isFinite(n)&&n>=.05&&n<=3){f.duration=n;save();}else duration.value=f.duration;render();};label.append(duration,document.createTextNode('秒'));
      const tools=document.createElement('div');tools.className='pose-tools';
      const edit=document.createElement('button');edit.textContent=editing===i?'保存':'编辑';edit.disabled=busy||(editing>=0&&editing!==i);edit.setAttribute('aria-label',`${editing===i?'保存':'编辑'}姿势 ${i+1}`);
      edit.onclick=async()=>{
        if(editing===i){if(!isReady()){text('正在更新构形，请稍后保存。');return;}const pose=getPose();frames[i]={...pose,params:[...pose.params],duration:f.duration,name:f.name||frameName(f),...(f.silent?{silent:true}:{})};try{await save();editing=-1;syncDirty();render();text('当前关键帧已保存。');}catch{frames[i]=f;}}
        else{editing=i;selected=i;applyPose(f);dirty(true);render();text('正在编辑 '+frameName(f)+'，可切换到器官或声音调整，再返回保存。');}
      };tools.append(edit);
      if(editing===i){const cancel=document.createElement('button');cancel.textContent='取消';cancel.onclick=()=>{editing=-1;applyPose(f);syncDirty();render();};tools.append(cancel);}
      for(const [name,fn,disabled] of [['←',()=>{[frames[i-1],frames[i]]=[frames[i],frames[i-1]];selected=i-1;},i===0],['→',()=>{[frames[i+1],frames[i]]=[frames[i],frames[i+1]];selected=i+1;},i===frames.length-1],['移除',()=>{frames.splice(i,1);selected=-1;},false]]){
        const b=document.createElement('button');b.textContent=name;b.disabled=busy||disabled||editing>=0;b.setAttribute('aria-label',`${name==='←'?'前移':name==='→'?'后移':'移除'}姿势 ${i+1}`);b.onclick=()=>{fn();save();render();};tools.append(b);
      }
      const timing=document.createElement('span');timing.className='pose-time';timing.textContent=`⠿ ${intervals[i].start.toFixed(2)}–${intervals[i].end.toFixed(2)} s`;timing.title='上下拖动排序';
      card.append(title,name,detail,timing,label,tools);$('keyframeList').append(card);
    });
    const total=Math.round(frames.reduce((s,f)=>s+f.duration,0)*48000)/48000;
    $('frameCount').textContent=frames.length;$('transportStatus').textContent=frames.length?`${frames.length} 个姿势 · ${total.toFixed(2)} 秒`:'保存多个姿势后连续播放';
    $('captureFrame').disabled=busy||editing>=0;$('captureSilence').disabled=busy||editing>=0;$('playFrames').disabled=busy||editing>=0||isSilent||frames.length<2||total>12;$('stopFrames').disabled=!busy;
    for(const id of ['importFrames','exportFrames'])$(id).disabled=busy||editing>=0;
    $('clearFrames').disabled=busy||editing>=0||!frames.length;
    $('undoClearFrames').hidden=!clearedSnapshot;$('undoClearFrames').disabled=busy||editing>=0;
    $('exportVideo').disabled=busy||editing>=0||frames.length<2||total>12;
    $('keyframeList').scrollTop=scroll;
    pitch.draw();
    if(!busy)text(editing>=0?'正在编辑 '+frameName(frames[editing])+' · 调整后点击此卡片的保存':frames.length?`${frames.length} 个姿势 · ${total.toFixed(2)} 秒${total>12?'，请缩短到 12 秒以内':''}`:'先摆出姿势，再保存；至少需要两个关键帧。');
  }
  function lock(value){busy=value;setBusy(value);$('motionHud').hidden=!value;render();}
  function capture(silent=false){if(!isReady()){text('正在更新构形，稍后再保存。');return;}const pose=getPose();frames.push({...pose,params:[...pose.params],duration:.2,...(silent?{silent:true,name:'静音'}:{})});selected=frames.length-1;save();render();$('keyframeList').lastElementChild?.scrollIntoView({block:'nearest'});}
  $('captureFrame').onclick=()=>capture();$('captureSilence').onclick=()=>capture(true);
  $('clearFrames').onclick=()=>{
    if(busy||editing>=0||!frames.length)return;
    clearedSnapshot={frames:JSON.parse(JSON.stringify(frames)),curve:pitch.getCurve(),selected};
    frames=[];selected=-1;pitch.setCurve([]);save(true);render();text('已清空关键帧和手绘基频，可点击撤销清空恢复。');
  };
  $('undoClearFrames').onclick=()=>{
    if(busy||editing>=0||!clearedSnapshot)return;
    const previous=clearedSnapshot;frames=previous.frames;selected=previous.selected;pitch.setCurve(previous.curve);
    save();render();text('已恢复清空前的关键帧和手绘基频。');
  };
  $('stopFrames').onclick=async()=>{++generation;try{await post('animation/stop');}catch{}lock(false);text('已停止，保留当前姿势。');};
  $('stopMotionOverlay').onclick=()=>$('stopFrames').click();
  async function prepare(){
    const prepared=await(await post('animation/prepare',{frames,pitch_curve:pitch.getCurve(),keep_vowel:keepVowel()})).json();
    const adjusted=JSON.stringify(prepared.frames.map(f=>[f.params,f.manual_root]))!==JSON.stringify(frames.map(f=>[f.params,f.manual_root]));
    if(adjusted){frames=prepared.frames;await save();render();}
    return {...prepared,adjusted};
  }
  function lockFiles(){lock(true);$('motionHud').hidden=true;$('stopFrames').disabled=true;}
  $('exportFrames').onclick=async()=>{
    if(busy||editing>=0)return;lockFiles();let message='';
    try{await saveQueue;const result=await(await post('document/save',{frames,pitch_curve:pitch.getCurve()})).json();if(result.saved)message='已导出 '+result.name;}
    catch(e){message='导出失败：'+e.message;}
    finally{lock(false);if(message)text(message);}
  };
  $('importFrames').onclick=async()=>{
    if(busy||editing>=0)return;lockFiles();let message='';
    try{
      await saveQueue;const result=await(await post('document/open')).json();if(result.cancelled)return;
      const incoming=result.document;await post('keyframes',{frames:incoming.frames,pitch_curve:incoming.pitch_curve});
      frames=incoming.frames;clearedSnapshot=null;pitch.setCurve(incoming.pitch_curve);selected=-1;message='已导入 '+result.name+' · '+frames.length+' 个姿势';
    }catch(e){message='导入失败，当前序列保留：'+e.message;}
    finally{lock(false);if(message)text(message);}
  };
  import('./video.js').then(({setupVideo})=>setupVideo({post,viewer,getFrames:()=>frames,getCurve:pitch.getCurve,prepare,lock,ready:()=>!busy&&editing<0&&isReady()})).catch(e=>text('视频组件载入失败：'+e.message));
  $('playFrames').onclick=async()=>{
    if(!isReady()){text('正在更新构形，稍后再播放。');return;}
    const run=++generation;lock(true);$('motionClock').textContent='正在生成声音与画面';text('正在生成声音与同步画面…');$('motionProgress').value=0;
    try{
      const prepared=await prepare();if(run!==generation)return;
      const adjusted=prepared.adjusted;
      $('playFrames').dataset.cache=prepared.cached?'hit':'miss';
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
          else{lock(false);text(frame.completed?'播放结束 · 已保留最后一个姿势'+(prepared.cached?' · 已复用上次动作与音频':''):'已停止');}
        }catch(e){await post('animation/stop').catch(()=>{});lock(false);text('播放失败：'+e.message);}
      };await tick();
    }catch(e){if(run===generation){lock(false);text('生成失败：'+e.message);}}
  };
  document.addEventListener('pitch-layout',render);
  render();return {isBusy:()=>busy};
}
