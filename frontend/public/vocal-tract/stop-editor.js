// Explicit phase controls share the ordinary keyframe list and saved poses.
const expanded=new Set();
const ms=value=>Number((value*1000).toFixed(3));
export function initialStopTiming(duration){
 const release=Math.min(.008,duration*.1),closure=Math.max(Math.min(.06,duration*.7),duration-release-1),aspiration=duration-closure-release;
 return {closure,release,aspiration,vot:release+aspiration};
}
export function editStopTiming(previous,key,milliseconds){
 const value=Number(milliseconds)/1000,t={...previous,[key]:value};
 if(String(milliseconds).trim()===''||!Number.isFinite(value))throw Error('请输入有效的毫秒数。');
 if(key==='aspiration')t.vot=t.release+t.aspiration;
 else t.aspiration=Math.max(0,t.vot-t.release);
 const duration=t.closure+t.release+t.aspiration;
 if(t.closure<.01||t.closure>2||t.release<.001||t.release>.05||t.aspiration<0||t.aspiration>1||duration>3)throw Error('闭塞 10–2000 ms，释放 1–50 ms，送气 0–1000 ms，总时长不超过 3 秒。');
 if(t.vot < -t.closure)throw Error('前浊不能早于闭塞开始，请增加闭塞时长或缩短负 VOT。');
 return t;
}
function timingHelp(text){
 const details=document.createElement('details');details.className='stop-help';const summary=document.createElement('summary');summary.textContent='时序说明';const paragraph=document.createElement('p');paragraph.className='minor';paragraph.textContent=text;details.append(summary,paragraph);return details;
}
export function stopEditor({frame,index,frames,busy,onChange}){
 const details=document.createElement('details');details.className='stop-timing';details.open=expanded.has(frame.frame_id);details.ontoggle=()=>{if(details.isConnected)details.open?expanded.add(frame.frame_id):expanded.delete(frame.frame_id);};
 const commit=()=>{if(details.open)expanded.add(frame.frame_id);onChange();};
 const summary=document.createElement('summary');summary.textContent=frame.stop_timing?'爆破时序 · 已启用':'爆破时序';details.append(summary);
 const label=document.createElement('label'),toggle=document.createElement('input');toggle.type='checkbox';toggle.checked=!!frame.stop_timing;toggle.disabled=busy;label.append(toggle,document.createTextNode('分开控制闭塞、释放与浊音起始'));details.append(label);
 const error=document.createElement('p');error.setAttribute('role','status');error.className='stop-error';details.append(error);
 toggle.onchange=()=>{if(toggle.checked){let duration=frame.duration,count=0;while(frames[index+1+count]?.aspiration){duration+=frames[index+1+count].duration;count++;}if(duration>3){toggle.checked=false;error.textContent='合并后超过单帧 3 秒，请先缩短旧送气段。';return;}frames.splice(index+1,count);frame.duration=duration;frame.stop_timing=initialStopTiming(duration);}else{if(frame.duration<.05){toggle.checked=true;error.textContent='普通帧至少 50 ms，请先增加闭塞时长再关闭时序。';return;}delete frame.stop_timing;}commit();};
 if(!frame.stop_timing){details.append(timingHelp('后续构型作为释放目标；启用时合并紧随其后的旧送气帧，原声源设置保留。'));return details;}
 for(const [key,title,min,max] of [['closure','闭塞',10,2000],['release','释放',1,50],['aspiration','送气',0,1000],['vot','VOT',-2000,1050]]){
  const row=document.createElement('label');row.textContent=title;const input=document.createElement('input');Object.assign(input,{type:'number',step:'any',min,max,value:ms(frame.stop_timing[key]),disabled:busy});input.setAttribute('aria-label',`${title}毫秒，姿势 ${index+1}`);row.append(input,document.createTextNode('ms'));details.append(row);
  input.onchange=()=>{try{const t=editStopTiming(frame.stop_timing,key,input.value);frame.stop_timing=t;frame.duration=t.closure+t.release+t.aspiration;commit();}catch(e){error.textContent=e.message;input.value=ms(frame.stop_timing[key]);}};
 }
 const t=frame.stop_timing,timeline=document.createElement('p');timeline.className='stop-landmarks';timeline.textContent=`本帧：释放 ${ms(t.closure)} ms · 浊音起始 ${ms(t.closure+t.vot)} ms · 结束 ${ms(frame.duration)} ms`;details.append(timeline);
 details.append(timingHelp('VOT 为释放至浊音起始的时间差。负值表示闭塞期前浊，正值表示浊音延迟。修改 VOT 会同步送气时长，修改送气会设为正 VOT。总时长为闭塞、释放与释放后送气之和。'));return details;
}
